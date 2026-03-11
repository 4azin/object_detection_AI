# (로컬 터미널에서 필요 패키지 설치: pip install ultralytics opencv-python torch torchvision pandas scikit-learn matplotlib)
# ==========================================
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging
import os
import shutil
import time
from sklearn.cluster import KMeans

# 로그 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 🛠️ 코어 모듈 정의 (추론 전 단계)
# ==========================================

# 수중 이미지 전처리 (CLAHE) - 품질 향상
def apply_clahe(frame_bgr: np.ndarray) -> np.ndarray:
    """수중 환경의 낮은 대비를 개선하는 전처리 과정"""
    # 컬러 스페이스 LAB으로 변환
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # L 채널(밝기 채널)에 대비 제한 적응형 히스토그램 평활화(CLAHE) 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # 병합 후 다시 BGR로 변환
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr

# 흔들림 감지 (수중 기준 완화: 15.0) -> 디스크 로드 제거 및 in-memory array 처리로 변경
def is_frame_stable(frame_bgr: np.ndarray, blur_threshold: float = 15.0) -> bool:
    if frame_bgr is None: return False
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= blur_threshold

# 비디오 프레임 추출 (1 FPS) -> In-Memory 기반으로 디스크 쓰기 병목 제거
def extract_frames_from_video(video_path: str, fps: int = 1) -> List[Dict]:
    logger.info(f"비디오 프레임 추출 시작 (In-Memory 방식): {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError("비디오 파일을 열 수 없습니다.")
    video_fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = video_fps // fps if video_fps > 0 else 30
    frames_data = []
    count, saved_count = 0, 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % frame_interval == 0:
            # 메모리단에서 흔들림 즉시 검사
            if is_frame_stable(frame):
                sec = count // video_fps if video_fps > 0 else count // 30
                timestamp = f"{sec//60:02d}:{sec%60:02d}"
                frames_data.append({
                    "sec": sec,
                    "timestamp": timestamp,
                    "frame": frame  # np.ndarray 원본 프레임 그대로 보관 (디스크 미저장)
                })
                saved_count += 1
        count += 1
    cap.release()
    logger.info(f"메모리 상에 프레임 추출 빛 검사 완료! 총 {saved_count}장 유효 버퍼 생성.")
    return frames_data

# [고도화] NIMA 모델 아키텍처 (ResNet50 기반)
class NIMA_ResNet50(nn.Module):
    def __init__(self):
        super(NIMA_ResNet50, self).__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
        
    def extract_features(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)

# ==========================================
# 🧠 추론 및 시각화 엔진
# ==========================================
class HighlightEngine:
    def __init__(self, nima_weights: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("YOLO-World Large(l) 모델 로딩 중... (객체 탐지 성능 극대화)")
        self.yolo_model = YOLO("yolov8l-world.pt")

        underwater_classes = [
            "colorful tropical fish", "sea turtle swimming", "manta ray", 
            "nudibranch", "large shark", "jellyfish", "octopus", 
            "beautiful coral reef", "scuba diver"
        ]
        self.yolo_model.set_classes(underwater_classes)
        logger.info(f"✅ 디테일 탐지 대상 설정 완료: {underwater_classes}")

        self.nima_model = NIMA_ResNet50().to(self.device)
        
        if nima_weights and Path(nima_weights).exists():
            try:
                self.nima_model.load_state_dict(torch.load(nima_weights, map_location=self.device))
                logger.info(f"✅ NIMA 가중치 로드 완료: {nima_weights}")
            except Exception as e:
                logger.warning(f"⚠️ 기존 가중치 로드 실패. 에러: {e}")
        else:
            logger.warning(f"⚠️ NIMA 가중치 경로를 찾을 수 없습니다. 기본 가중치로 진행합니다.")
            
        self.nima_model.eval()

        self.nima_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _calculate_scores(self, yolo_result, nima_prob, frame_width, frame_height):
        y_score = 0.0
        detected = []
        
        # 1. YOLO 스코어 계산 (객체 크기 가중치 + Center Bias 추가)
        if len(yolo_result.boxes) > 0:
            frame_area = frame_width * frame_height
            center_x, center_y = frame_width / 2.0, frame_height / 2.0
            max_dist = np.sqrt(center_x**2 + center_y**2)  # 중심에서 모서리까지의 대각선 거리
            
            scores = []
            
            for box in yolo_result.boxes:
                obj_name = yolo_result.names[int(box.cls.item())]
                detected.append(obj_name)
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_area = (x2 - x1) * (y2 - y1)
                area_ratio = box_area / frame_area # 0.0 ~ 1.0 비율
                
                # [고도화 1: Size Weight Clamping] 상한선 설정
                if area_ratio > 0.4:  # 화면의 40% 이상을 차지하는 거대한 피사체
                    size_weight = 0.6  # 초근접/형태불분명 등으로 간주하고 패널티 부여 (Clamp)
                else:
                    size_weight = 1.0 + (area_ratio * 1.5)  # 보통 피사체의 경우 면적에 비례 가산
                
                # [고도화 2: Center Bias] 화면 중앙 보너스
                obj_center_x = (x1 + x2) / 2.0
                obj_center_y = (y1 + y2) / 2.0
                
                # 피사체의 중심과 프레임 중앙 사이의 거리 계산
                dist_from_center = np.sqrt((obj_center_x - center_x)**2 + (obj_center_y - center_y)**2)
                
                # Center Ratio: 화면 중앙일수록 1에 가깝고, 모서리일수록 0에 가까워짐
                center_ratio = max(0.0, 1.0 - (dist_from_center / max_dist))
                
                # 중앙에 가까울수록 최대 1.4배의 가중치, 외곽에 있으면 0.8까지 하락
                center_bias_weight = 0.8 + (center_ratio * 0.6) 
                
                base_conf = box.conf.item()
                # 모든 가중치 융합: 기본정확도 * 크기가중치 * 중심가중치
                weighted_score = base_conf * size_weight * center_bias_weight
                scores.append(weighted_score)
                
            y_score = float(np.max(scores))

        # 2. NIMA 스코어 계산
        scores_tensor = torch.arange(1, 11).float().to(self.device)
        mean_score = torch.sum(nima_prob * scores_tensor, dim=1).item()
        n_score = (mean_score - 1) / 9.0

        # 3. 점수 융합
        tau = 0.15 
        if y_score >= tau:
            return (0.6 * y_score) + (0.4 * n_score), "Prominent Marine Life", list(set(detected))
        return 0.5 * n_score, "Aesthetic Background", list(set(detected))

    def process_video_and_rank(self, frames_data: List[Dict], k: int = 5) -> List[Dict]:
        results_meta = []
        features_list = []
        frame_store = {} # 메모리 점유 최적화 및 pandas 오류 방지를 위해 프레임 배열 분리
        batch_size = 8
        
        logger.info("통합 추론 및 특징 추출 시작 (In-Memory, CLAHE 전처리 적용)...")

        for i in range(0, len(frames_data), batch_size):
            batch_data = frames_data[i:i+batch_size]
            if not batch_data: continue

            # CLAHE 수중 조명/대비 전처리 적용 (인식률 향상 목적)
            clahe_frames_bgr = [apply_clahe(item['frame']) for item in batch_data]
            
            # NIMA는 PIL RGB Image를 요구하므로 형태 변환
            pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in clahe_frames_bgr]
            
            # YOLO는 list of BGR numpy arrays를 직접 받을 수 있음
            yolo_preds = self.yolo_model.predict(clahe_frames_bgr, imgsz=640, conf=0.1, verbose=False)
            
            # NIMA 텐서화
            nima_tensors = torch.stack([self.nima_transform(img) for img in pil_images]).to(self.device)

            with torch.no_grad():
                nima_probs = self.nima_model(nima_tensors)
                feats_flat = self.nima_model.extract_features(nima_tensors).cpu().numpy()

            for j, item in enumerate(batch_data):
                frame_h, frame_w = clahe_frames_bgr[j].shape[:2]
                f_score, cat, obj = self._calculate_scores(yolo_preds[j], nima_probs[j].unsqueeze(0), frame_w, frame_h)
                
                sec_id = item['sec']
                # 원본 이미지를 저장 (화면에 보여줄 시 CLAHE 변환 전/후 고민가능하나, 여기선 원본의 자연스러운 모습 반환)
                frame_store[sec_id] = item['frame']

                results_meta.append({
                    "timestamp": item['timestamp'], "sec": sec_id,
                    "final_score": round(f_score, 4), "category": cat, "objects": obj
                })
                features_list.append(feats_flat[j])

        df = pd.DataFrame(results_meta)
        if df.empty: return []
        
        if len(df) <= k:
            selected_records = df.sort_values(by="final_score", ascending=False).to_dict('records')
            for res in selected_records:
                res['frame_bgr'] = frame_store[res['sec']]
            return selected_records

        # K-Means 시각적 군집화 및 다양성 확보
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(np.array(features_list))

        final_selected = []
        selected_objects_pool = set()
        cluster_max_scores = df.groupby('cluster')['final_score'].max().sort_values(ascending=False)

        for cluster_id in cluster_max_scores.index:
            cluster_df = df[df['cluster'] == cluster_id].sort_values(by="final_score", ascending=False)
            best_frame = None
            top_score = cluster_df.iloc[0]['final_score']

            for _, row in cluster_df.iterrows():
                if row['final_score'] >= top_score * 0.8:
                    current_objects = set(row['objects'])
                    if current_objects and not current_objects.issubset(selected_objects_pool):
                        best_frame = row
                        break

            if best_frame is None:
                best_frame = cluster_df.iloc[0]

            selected_objects_pool.update(best_frame['objects'])

            frame_dict = best_frame.to_dict()
            frame_dict['frame_bgr'] = frame_store[frame_dict['sec']]
            final_selected.append(frame_dict)

        return final_selected

# ==========================================
# 🚀 실행 파이프라인
# ==========================================
if __name__ == "__main__":
    MY_DIVING_VIDEO = "./samples/ul1.mp4" 
    FINAL_HIGHLIGHTS_DIR = "./final_highlights"

    NIMA_WEIGHTS_PATH = "./weights/nima_resnet50_finetuned.pth"

    if os.path.exists(MY_DIVING_VIDEO):
        if os.path.exists(FINAL_HIGHLIGHTS_DIR): shutil.rmtree(FINAL_HIGHLIGHTS_DIR)
        Path(FINAL_HIGHLIGHTS_DIR).mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()

        # 1. 프레임 추출 (메모리 제너레이션)
        extracted_frames_in_memory = extract_frames_from_video(MY_DIVING_VIDEO, fps=1)

        # 2. 엔진 가동 및 분석
        engine = HighlightEngine(nima_weights=NIMA_WEIGHTS_PATH)
        top_highlights = engine.process_video_and_rank(extracted_frames_in_memory, k=5)

        print("\n" + "="*60)
        print("🎬 [개편 완료: In-Memory / CLAHE / Center Bias & Size Clamp] 🎬")
        print("="*60 + "\n")

        # 3. 디스크 저장 (최종으로 승리한 5장만 I/O 수행)
        for idx, res in enumerate(top_highlights, 1):
            clean_ts = res['timestamp'].replace(":", "")
            new_file_name = f"rank{idx}_{clean_ts}_score{res['final_score']}.jpg"
            save_path = os.path.join(FINAL_HIGHLIGHTS_DIR, new_file_name)

            # 디스크 최종 기록
            cv2.imwrite(save_path, res['frame_bgr'], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            print(f"[{idx}위 프레임 저장 완료] {save_path}")
            print(f" ↳ 판독 점수: {res['final_score']} | 영상 타임스탬프: {res['timestamp']}")
            print(f" ↳ 탐지 객체: {res['objects']}\n")

        total_time = time.time() - start_time
        print(f"최종 하이라이트 처리가 매우 빠르게 완료되었습니다. (결과 폴더: {FINAL_HIGHLIGHTS_DIR})")
        print(f"⏱️ 총 소요 시간: {total_time:.2f} 초")
    else:
        print(f"[{MY_DIVING_VIDEO}] 파일을 찾을 수 없습니다. 경로를 확인해주세요.")