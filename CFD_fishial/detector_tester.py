"""
Divery Vision Pipeline - Module A: Detector 독립 테스트
=====================================================
CFD(Custom Fish Detector) 가중치를 사용한 물고기 객체 탐지 성능 측정 모듈.
ultralytics YOLOv12x 기반, FP16 추론, RTX 4070 최적화.
"""

import csv
import time
from pathlib import Path
from typing import Any

import cv2
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO


class DetectorTester:
    """
    CFD 가중치 기반 물고기 객체 탐지 독립 테스트 클래스.

    주요 기능:
        - ultralytics YOLO 모델 로드 (FP16 Half-precision)
        - 영상 파일 프레임별 추론 및 지표 수집
        - 탐지 결과 렌더링 영상 저장
        - detection_log.csv 출력
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """
        설정 파일을 로드하고 YOLO 모델을 초기화한다.

        Args:
            config_path: YAML 설정 파일 경로.
        """
        self.config = self._load_config(config_path)
        self.device = self._select_device()
        self.model = self._load_model()

        # 출력 디렉토리 생성
        self.output_dir = Path(self.config["output"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[DetectorTester] 초기화 완료")
        print(f"  - Device : {self.device}")
        print(f"  - Weights: {self.config['model']['weights']}")
        print(f"  - FP16   : {self.config['model']['half']}")
        print(f"  - Output : {self.output_dir.resolve()}")

    # ------------------------------------------------------------------
    # Initialization Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """YAML 설정 파일을 읽어 딕셔너리로 반환."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _select_device() -> str:
        """CUDA 사용 가능 여부를 확인하고 디바이스를 반환."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[DetectorTester] GPU 감지: {gpu_name}")
            return "cuda:0"
        print("[DetectorTester] [WARNING] GPU를 사용할 수 없습니다. CPU로 실행합니다.")
        return "cpu"

    def _load_model(self) -> YOLO:
        """ultralytics YOLO 모델을 로드하고 디바이스에 할당."""
        weights_path = Path(self.config["model"]["weights"])
        if not weights_path.exists():
            raise FileNotFoundError(
                f"가중치 파일을 찾을 수 없습니다: {weights_path.resolve()}"
            )

        print(f"[DetectorTester] 모델 로드 중: {weights_path} ...")
        model = YOLO(str(weights_path))
        return model

    # ------------------------------------------------------------------
    # Core: Single Video Processing
    # ------------------------------------------------------------------

    def run(self, video_path: str) -> list[dict]:
        """
        단일 영상 파일에 대해 Detector 테스트를 수행한다.

        Args:
            video_path: 테스트할 영상 파일 경로.

        Returns:
            프레임별 탐지 결과 딕셔너리 리스트.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"영상 파일을 열 수 없습니다: {video_path}")

        # 영상 메타데이터
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_original = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n{'='*60}")
        print(f"[영상 처리 시작] {video_path.name}")
        print(f"  해상도: {width}x{height} | 원본 FPS: {fps_original:.1f} | 총 프레임: {total_frames}")
        print(f"{'='*60}")

        # 결과 렌더링 영상 Writer 설정
        video_writer = None
        if self.config["output"]["save_video"]:
            out_video_path = self.output_dir / f"detected_{video_path.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(out_video_path), fourcc, fps_original, (width, height)
            )

        # 프레임별 처리
        detection_log: list[dict] = []
        frame_id = 0

        with tqdm(total=total_frames, desc=f"[CFD] {video_path.name}", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 추론 수행 및 지표 수집
                result, metrics = self._process_frame(frame)

                # 로그 기록
                log_entry = {
                    "Frame_ID": frame_id,
                    "Object_Count": metrics["object_count"],
                    "Avg_Confidence": round(metrics["avg_confidence"], 4),
                    "Inference_Time(ms)": round(metrics["inference_time_ms"], 2),
                    "FPS": round(metrics["fps"], 2),
                }
                detection_log.append(log_entry)

                # 결과 렌더링 영상 쓰기
                if video_writer is not None:
                    rendered = self._render_frame(frame, result)
                    video_writer.write(rendered)

                frame_id += 1
                pbar.set_postfix(
                    objs=metrics["object_count"],
                    fps=f"{metrics['fps']:.1f}",
                    conf=f"{metrics['avg_confidence']:.2f}",
                )
                pbar.update(1)

        # 리소스 해제
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"[저장 완료] 렌더링 영상: {out_video_path}")

        # CSV 저장
        csv_path = self.output_dir / self.config["output"]["csv_filename"]
        self._save_csv(detection_log, csv_path, video_path.name)

        # 요약 통계 출력
        self._print_summary(detection_log, video_path.name)

        return detection_log

    # ------------------------------------------------------------------
    # Frame-Level Processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame) -> tuple[Any, dict]:
        """
        단일 프레임에 대해 YOLO 추론을 수행하고 지표를 계산한다.

        Args:
            frame: OpenCV BGR 프레임 (numpy array).

        Returns:
            (ultralytics_result, metrics_dict) 튜플.
        """
        model_cfg = self.config["model"]

        # 추론 시간 측정 (GPU 동기화 포함)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=model_cfg["confidence_threshold"],
            iou=model_cfg["iou_threshold"],
            imgsz=model_cfg["img_size"],
            half=model_cfg["half"],
            device=self.device,
            verbose=False,
        )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        result = results[0]
        inference_time_ms = (t_end - t_start) * 1000

        # 지표 계산
        boxes = result.boxes
        object_count = len(boxes)
        avg_confidence = (
            float(boxes.conf.mean()) if object_count > 0 else 0.0
        )
        fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0.0

        metrics = {
            "object_count": object_count,
            "avg_confidence": avg_confidence,
            "inference_time_ms": inference_time_ms,
            "fps": fps,
        }
        return result, metrics

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_frame(frame, result) -> "cv2.Mat":
        """
        프레임에 탐지 결과(바운딩 박스 + 라벨)를 오버레이한다.

        Args:
            frame: 원본 BGR 프레임.
            result: ultralytics 추론 결과 객체.

        Returns:
            렌더링된 프레임.
        """
        rendered = result.plot(img=frame.copy())
        return rendered

    # ------------------------------------------------------------------
    # CSV Output
    # ------------------------------------------------------------------

    @staticmethod
    def _save_csv(log: list[dict], csv_path: Path, video_name: str) -> None:
        """
        탐지 로그를 CSV 파일로 저장한다.

        Args:
            log: 프레임별 탐지 결과 딕셔너리 리스트.
            csv_path: 저장할 CSV 파일 경로.
            video_name: 소스 영상 파일명 (로그 출력용).
        """
        if not log:
            print(f"[경고] {video_name}: 탐지 로그가 비어있어 CSV를 생성하지 않습니다.")
            return

        fieldnames = [
            "Frame_ID",
            "Object_Count",
            "Avg_Confidence",
            "Inference_Time(ms)",
            "FPS",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log)

        print(f"[저장 완료] CSV 로그: {csv_path}")

    # ------------------------------------------------------------------
    # Summary Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(log: list[dict], video_name: str) -> None:
        """테스트 결과 요약 통계를 콘솔에 출력한다."""
        if not log:
            return

        total_frames = len(log)
        total_objects = sum(entry["Object_Count"] for entry in log)
        avg_objects = total_objects / total_frames
        avg_conf = (
            sum(entry["Avg_Confidence"] for entry in log) / total_frames
        )
        avg_inf_time = (
            sum(entry["Inference_Time(ms)"] for entry in log) / total_frames
        )
        avg_fps = sum(entry["FPS"] for entry in log) / total_frames

        print(f"\n{'='*60}")
        print(f"[SUMMARY] 테스트 결과 요약: {video_name}")
        print(f"{'='*60}")
        print(f"  총 프레임 수        : {total_frames}")
        print(f"  총 탐지 객체 수     : {total_objects}")
        print(f"  프레임당 평균 객체  : {avg_objects:.2f}")
        print(f"  평균 Confidence     : {avg_conf:.4f}")
        print(f"  평균 추론 시간      : {avg_inf_time:.2f} ms")
        print(f"  평균 FPS            : {avg_fps:.1f}")
        print(f"  실시간 처리(>=30FPS): {'[O] 가능' if avg_fps >= 30 else '[X] 불가'}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """GPU 메모리를 명시적으로 해제한다."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[DetectorTester] GPU 메모리 해제 완료.")
