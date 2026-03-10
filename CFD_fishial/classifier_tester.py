"""
Divery Vision Pipeline - Module B: Fishial.AI Classifier Tester
===============================================================
Fishial.AI v10.0 (755 Classes) 임베딩 기반 어종 분류기 독립 테스트 모듈.
ArcFace + kNN(FAISS) 하이브리드 추론 방식 사용.
"""

import csv
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

# fishial_model/ 내 inference.py를 직접 임포트
sys.path.insert(0, str(Path(__file__).parent / "fishial_model"))
from inference import EmbeddingClassifier  # noqa: E402


class ClassifierTester:
    """
    Fishial.AI 임베딩 분류기 독립 테스트 클래스.

    주요 기능:
        - Fishial.AI v10.0 모델 (model.ckpt + database.pt) 로드
        - 이미지 폴더 내 물고기 크롭 이미지를 순회하며 Top-3 종 분류
        - classification_log.csv 출력
        - GPU 메모리 관리
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = self._load_config(config_path)
        cls_cfg = self.config["classifier"]

        # EmbeddingClassifier config 빌드
        self.ec_config = self._build_ec_config(cls_cfg)

        print("[ClassifierTester] Loading Fishial.AI model...")
        self.classifier = EmbeddingClassifier(self.ec_config)

        # Warmup for stable GPU timing
        print("[ClassifierTester] GPU warmup...")
        self.classifier.warmup(num_iterations=3)

        self.topk = cls_cfg.get("topk_results", 3)
        self.output_dir = Path(self.config["output"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ClassifierTester] Initialized")
        print(f"  - Classes  : {cls_cfg['num_classes']}")
        print(f"  - Device   : {cls_cfg['device']}")
        print(f"  - Top-K    : {self.topk}")
        print(f"  - kNN      : {cls_cfg.get('use_knn', True)}")

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
    def _build_ec_config(cls_cfg: dict) -> dict:
        """ClassifierTester용 config를 EmbeddingClassifier용 dict로 변환."""
        return {
            "log_level": "INFO",
            "dataset": {
                "path": cls_cfg["database_path"],
            },
            "model": {
                "checkpoint_path": cls_cfg["checkpoint_path"],
                "backbone_model_name": cls_cfg.get("backbone_model_name", "maxvit_base_tf_224"),
                "embedding_dim": cls_cfg.get("embedding_dim", 512),
                "num_classes": cls_cfg.get("num_classes", 755),
                "arcface_s": cls_cfg.get("arcface_s", 64.0),
                "arcface_m": cls_cfg.get("arcface_m", 0.2),
                "pooling_type": cls_cfg.get("pooling_type", "attention"),
                "input_size": cls_cfg.get("input_size", 224),
                "device": cls_cfg.get("device", "cuda:0"),
            },
            "use_knn": cls_cfg.get("use_knn", True),
        }

    # ------------------------------------------------------------------
    # Core: Classification
    # ------------------------------------------------------------------

    def run(self, image_dir: str) -> list[dict]:
        """
        이미지 폴더 내 모든 이미지에 대해 종 분류를 수행한다.

        Args:
            image_dir: 이미지 폴더 경로 (best_shots/ 등).

        Returns:
            분류 결과 딕셔너리 리스트.
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {image_dir}")

        # 지원 확장자 이미지 수집
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted(
            f for f in image_dir.iterdir()
            if f.suffix.lower() in extensions
        )

        if not image_files:
            print(f"[WARNING] No images found in: {image_dir}")
            return []

        print(f"\n{'='*60}")
        print(f"[Classification] Fishial.AI v10.0")
        print(f"  Images: {len(image_files)} | Top-K: {self.topk}")
        print(f"{'='*60}")

        results: list[dict] = []

        with tqdm(image_files, desc="[Classify]", unit="img") as pbar:
            for img_path in pbar:
                result = self._classify_single(img_path)
                if result is not None:
                    results.append(result)
                    pbar.set_postfix(
                        top1=result.get("Top1_Species", "?")[:15],
                        conf=f"{result.get('Top1_Prob', 0):.3f}",
                    )

        # CSV 저장
        csv_name = self.config["classifier"].get(
            "csv_filename", "classification_log.csv"
        )
        csv_path = self.output_dir / csv_name
        self._save_csv(results, csv_path)

        self._print_summary(results)
        return results

    # ------------------------------------------------------------------
    # Single image classification
    # ------------------------------------------------------------------

    def _classify_single(self, img_path: Path) -> Optional[dict]:
        """
        단일 이미지에 대해 Top-K 분류를 수행한다.

        Returns:
            결과 딕셔너리 or None (실패 시).
        """
        try:
            # BGR -> RGB
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  [SKIP] Cannot read: {img_path.name}")
                return None
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 추론 시간 측정
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            predictions = self.classifier(img_rgb)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            inference_ms = (t_end - t_start) * 1000

            # Top-K 결과 빌드
            row: dict = {"Image_Name": img_path.name}

            for k in range(self.topk):
                rank = k + 1
                if k < len(predictions):
                    pred = predictions[k]
                    row[f"Top{rank}_Species"] = pred.name
                    row[f"Top{rank}_Prob"] = round(pred.accuracy, 4)
                else:
                    row[f"Top{rank}_Species"] = ""
                    row[f"Top{rank}_Prob"] = 0.0

            row["Inference_Time(ms)"] = round(inference_ms, 2)
            return row

        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            return None

    # ------------------------------------------------------------------
    # CSV Output
    # ------------------------------------------------------------------

    def _save_csv(self, results: list[dict], csv_path: Path) -> None:
        """분류 결과를 CSV로 저장한다."""
        if not results:
            print("[WARNING] No results to save.")
            return

        fieldnames = ["Image_Name"]
        for k in range(1, self.topk + 1):
            fieldnames += [f"Top{k}_Species", f"Top{k}_Prob"]
        fieldnames.append("Inference_Time(ms)")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"[Save] Classification log -> {csv_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(results: list[dict]) -> None:
        if not results:
            return

        n = len(results)
        probs = [r.get("Top1_Prob", 0) for r in results]
        times = [r.get("Inference_Time(ms)", 0) for r in results]

        avg_prob = sum(probs) / n
        avg_time = sum(times) / n

        # 가장 자주 등장하는 Top-1 종
        from collections import Counter
        species_counts = Counter(r.get("Top1_Species", "") for r in results)
        top_species = species_counts.most_common(5)

        print(f"\n{'='*60}")
        print(f"[SUMMARY] Classification Results")
        print(f"{'='*60}")
        print(f"  Total images       : {n}")
        print(f"  Avg Top1 Prob      : {avg_prob:.4f}")
        print(f"  Avg inference time : {avg_time:.2f} ms")
        print(f"  Top-5 predicted species:")
        for species, count in top_species:
            pct = count / n * 100
            print(f"    - {species}: {count} ({pct:.1f}%)")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """GPU 메모리 및 모델 리소스 해제."""
        if hasattr(self, "classifier"):
            self.classifier.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[ClassifierTester] GPU memory released.")
