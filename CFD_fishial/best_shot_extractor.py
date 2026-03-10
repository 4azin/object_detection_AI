"""
Divery Vision Pipeline - Module A-2: ByteTrack Best-Shot Extractor
==================================================================
CFD 가중치 + ByteTrack 추적으로 영상 내 물고기별 최고 Confidence 프레임을 추출.
각 track_id 당 가장 선명한 1장(Best Shot)을 저장한다.
"""

import csv
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO

from deduplicator import EmbeddingDeduplicator, FinalDeduplicator


class BestShotExtractor:
    """
    ByteTrack 기반 물고기 Best-Shot 추출 클래스.

    주요 기능:
        - YOLO + ByteTrack 추적으로 물고기별 고유 ID 부여
        - 각 ID별 최고 Confidence 프레임의 크롭을 메모리에 유지
        - 영상 처리 완료 후 best_shots/ 폴더에 ID별 이미지 저장
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = self._load_config(config_path)
        self.device = self._select_device()
        self.model = self._load_model()

        # Best shots 저장 경로
        self.best_shots_dir = Path(
            self.config["tracker"].get("best_shots_dir", "./results/best_shots")
        )
        self.best_shots_dir.mkdir(parents=True, exist_ok=True)

        # 결과 출력 디렉토리
        self.output_dir = Path(self.config["output"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Deduplicator (lazy init - 분류 모델 로딩 시점 제어)
        self._deduplicator: Optional[EmbeddingDeduplicator] = None
        self._final_dedup: Optional[FinalDeduplicator] = None

        safeguard = self.config.get("safeguard", {})
        self._emb_sim_thresh = safeguard.get("embedding_similarity_threshold", 0.85)
        self._merge_frame_gap = safeguard.get("merge_frame_gap", 150)
        self._merge_sim_thresh = safeguard.get("merge_similarity_threshold", 0.80)

        print(f"[BestShotExtractor] Initialized")
        print(f"  - Device : {self.device}")
        print(f"  - Tracker: {self.config['tracker']['tracker_type']}")
        print(f"  - Dedup  : emb_thresh={self._emb_sim_thresh}, merge_gap={self._merge_frame_gap}")
        print(f"  - Output : {self.best_shots_dir.resolve()}")

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str) -> dict:
        cfg = Path(config_path)
        if not cfg.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(cfg, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            print(f"[BestShotExtractor] GPU: {gpu}")
            return "0"  # ultralytics uses device index string
        print("[BestShotExtractor] [WARNING] No GPU. Running on CPU.")
        return "cpu"

    def _load_model(self) -> YOLO:
        weights = Path(self.config["model"]["weights"])
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights.resolve()}")
        print(f"[BestShotExtractor] Loading model: {weights}")
        return YOLO(str(weights))

    # ------------------------------------------------------------------
    # Core: Best-Shot Extraction
    # ------------------------------------------------------------------

    def run(self, video_path: str) -> dict:
        """
        영상에서 ByteTrack 추적 후 ID별 Best Shot을 추출한다.

        Args:
            video_path: 영상 파일 경로.

        Returns:
            {track_id: {"conf": float, "path": str}} 딕셔너리.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_orig = cap.get(cv2.CAP_PROP_FPS)
        cap.release()  # model.track will open internally

        print(f"\n{'='*60}")
        print(f"[Best-Shot Extraction] {video_path.name}")
        print(f"  frames: {total_frames} | fps: {fps_orig:.1f}")
        print(f"{'='*60}")

        tracker_cfg = self.config["tracker"]
        model_cfg = self.config["model"]

        # {track_id: {"conf": float, "crop": np.array, "frame_id": int}}
        best_shots: dict = {}
        frame_id = 0

        # model.track() yields results per frame as a generator
        results = self.model.track(
            source=str(video_path),
            persist=True,
            tracker=tracker_cfg["tracker_type"],
            conf=tracker_cfg["confidence_threshold"],
            iou=model_cfg["iou_threshold"],
            imgsz=model_cfg["img_size"],
            half=model_cfg["half"],
            device=self.device,
            stream=True,  # generator mode for memory efficiency
            verbose=False,
        )

        with tqdm(total=total_frames, desc=f"[Track] {video_path.name}", unit="frame") as pbar:
            for result in results:
                frame = result.orig_img
                boxes = result.boxes

                if boxes is not None and boxes.id is not None:
                    for i in range(len(boxes)):
                        tid = int(boxes.id[i].item())
                        conf = float(boxes.conf[i].item())
                        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)

                        # 현재 ID의 기존 best보다 conf가 높으면 갱신
                        if tid not in best_shots or conf > best_shots[tid]["conf"]:
                            x1, y1, x2, y2 = xyxy
                            # 경계 보정
                            h, w = frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)

                            if x2 > x1 and y2 > y1:
                                crop = frame[y1:y2, x1:x2].copy()
                                best_shots[tid] = {
                                    "conf": conf,
                                    "crop": crop,
                                    "frame_id": frame_id,
                                }

                n_tracked = len(best_shots)
                pbar.set_postfix(tracked=n_tracked)
                pbar.update(1)
                frame_id += 1

        # --- Safeguard 1: Embedding-based deduplication ---
        print(f"\n[Safeguard] Embedding-based deduplication...")
        dedup = self._get_deduplicator()
        filtered, removed = dedup.deduplicate_best_shots(best_shots)

        # 저장
        saved = self._save_best_shots(filtered, video_path.stem, fps_orig)

        # --- Safeguard 2: Final ID merge ---
        print(f"[Safeguard] Final ID merge...")
        final_dedup = self._get_final_dedup()
        merged = final_dedup.merge_overlapping_ids(saved)

        # 요약 CSV 저장
        csv_path = self.output_dir / f"best_shots_{video_path.stem}.csv"
        self._save_summary_csv(merged, csv_path)

        self._print_summary(merged, video_path.name, len(best_shots), len(removed))
        return merged

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_best_shots(self, best_shots: dict, video_stem: str, fps: float = 30.0) -> dict:
        """Best Shot 이미지를 파일로 저장한다."""
        saved: dict = {}
        for tid, data in best_shots.items():
            seconds = int(data["frame_id"] / fps) if fps > 0 else 0
            filename = f"{video_stem}_ID{tid:04d}_{data['conf']:.2f}_{seconds:06d}.jpg"
            filepath = self.best_shots_dir / filename
            cv2.imwrite(str(filepath), data["crop"])
            saved[tid] = {
                "conf": data["conf"],
                "frame_id": data["frame_id"],
                "path": str(filepath),
            }
        print(f"[Save] {len(saved)} best shots -> {self.best_shots_dir}")
        return saved

    @staticmethod
    def _save_summary_csv(saved: dict, csv_path: Path) -> None:
        """Best Shot 요약 CSV를 저장한다."""
        if not saved:
            return
        fieldnames = ["Track_ID", "Confidence", "Frame_ID", "Image_Path"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for tid, data in sorted(saved.items()):
                writer.writerow({
                    "Track_ID": tid,
                    "Confidence": round(data["conf"], 4),
                    "Frame_ID": data["frame_id"],
                    "Image_Path": data["path"],
                })
        print(f"[Save] Summary CSV -> {csv_path}")

    # ------------------------------------------------------------------
    # Deduplicator access
    # ------------------------------------------------------------------

    def _get_deduplicator(self) -> EmbeddingDeduplicator:
        if self._deduplicator is None:
            self._deduplicator = EmbeddingDeduplicator(
                config_path="config.yaml",
                similarity_threshold=self._emb_sim_thresh,
            )
        return self._deduplicator

    def _get_final_dedup(self) -> FinalDeduplicator:
        if self._final_dedup is None:
            self._final_dedup = FinalDeduplicator(
                frame_gap_threshold=self._merge_frame_gap,
                similarity_threshold=self._merge_sim_thresh,
            )
        return self._final_dedup

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(saved: dict, video_name: str, n_raw: int = 0, n_dedup_removed: int = 0) -> None:
        if not saved:
            print("[WARNING] No tracked objects found.")
            return

        confs = [d["conf"] for d in saved.values()]
        print(f"\n{'='*60}")
        print(f"[SUMMARY] Best-Shot Extraction: {video_name}")
        print(f"{'='*60}")
        if n_raw > 0:
            print(f"  Raw tracked IDs    : {n_raw}")
            print(f"  Dedup removed      : {n_dedup_removed}")
            print(f"  Merged removed     : {n_raw - n_dedup_removed - len(saved)}")
        print(f"  Final unique shots : {len(saved)}")
        print(f"  Avg confidence     : {sum(confs)/len(confs):.4f}")
        print(f"  Max confidence     : {max(confs):.4f}")
        print(f"  Min confidence     : {min(confs):.4f}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """GPU 메모리 해제."""
        if self._deduplicator is not None:
            self._deduplicator.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[BestShotExtractor] GPU memory released.")
