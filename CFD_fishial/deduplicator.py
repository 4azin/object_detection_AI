"""
Divery Vision Pipeline - Safeguard: 중복 제거 모듈
===================================================
BioCLIP-2 임베딩 기반으로 동일 개체의 반복 스냅샷을 제거한다.

  EmbeddingDeduplicator : 저장 전 실시간 중복 필터링
  FinalDeduplicator     : 저장 후 ID 병합 (후처리)
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms


class EmbeddingDeduplicator:
    """
    BioCLIP-2 임베딩 기반 중복 제거기.

    저장 전 각 best-shot 크롭의 임베딩을 추출하고,
    기존 저장된 이미지들과 cosine similarity를 비교하여
    중복(동일 개체의 다른 스냅샷)을 필터링한다.
    """

    def __init__(
        self,
        classifier: Optional[any] = None,
        config_path: str = "config.yaml",
        similarity_threshold: float = 0.85,
    ) -> None:
        self.threshold = similarity_threshold

        # 모델 로드 (외부에서 공유 가능) - 예: ClassifierTester 인스턴스
        if classifier is not None:
            self.model = classifier.model
            self.preprocess = classifier.preprocess
            self.device = classifier.device
            self._owns_classifier = False
        else:
            config = self._load_config(config_path)
            cls_cfg = config.get("classifier", {})
            try:
                from open_clip import create_model
            except ImportError:
                print("❌ open_clip_torch is not installed.")
                sys.exit(1)

            self.device = torch.device(cls_cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
            self.model_str = cls_cfg.get("model_str", "hf-hub:imageomics/bioclip-2")
            print(f"[Deduplicator] Loading {self.model_str} for embeddings...")
            
            self.model = create_model(self.model_str, output_dict=True, require_pretrained=True).to(self.device)
            self.model.eval()

            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ])
            self._owns_classifier = True

        print(f"[Deduplicator] Initialized (threshold={self.threshold})")

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def compute_embedding(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        BGR 이미지에서 L2-normalized 임베딩을 추출한다. (BioCLIP-2는 768차원)

        Args:
            img_bgr: BGR numpy array.

        Returns:
            float32 numpy array.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # torchvision transform
        pil_img = Image.fromarray(img_rgb)
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_features = self.model.encode_image(tensor)
            img_features = F.normalize(img_features, dim=-1)

        return img_features.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """두 L2-정규화 벡터 간 코사인 유사도."""
        return float(np.dot(a, b))

    def is_duplicate(
        self,
        new_emb: np.ndarray,
        existing_embs: list[np.ndarray],
    ) -> tuple[bool, float]:
        """
        새 임베딩이 기존 임베딩들과 중복인지 판정한다.

        Returns:
            (is_dup, max_similarity)
        """
        if not existing_embs:
            return False, 0.0

        sims = [self.cosine_similarity(new_emb, e) for e in existing_embs]
        max_sim = max(sims)
        return max_sim >= self.threshold, max_sim

    # ------------------------------------------------------------------
    # Batch deduplication
    # ------------------------------------------------------------------

    def deduplicate_best_shots(
        self, best_shots: dict
    ) -> tuple[dict, dict]:
        """
        Best-shot 딕셔너리에서 임베딩 기반 중복을 제거한다.

        Args:
            best_shots: {track_id: {"conf": float, "crop": np.array, "frame_id": int}}

        Returns:
            (filtered_shots, removed_info)
        """
        if not best_shots:
            return best_shots, {}

        # confidence 내림차순 정렬 (높은 conf 우선 유지)
        sorted_ids = sorted(
            best_shots.keys(),
            key=lambda tid: best_shots[tid]["conf"],
            reverse=True,
        )

        kept_embeddings: list[np.ndarray] = []
        kept_ids: list[int] = []
        filtered: dict = {}
        removed: dict = {}

        for tid in sorted_ids:
            data = best_shots[tid]
            emb = self.compute_embedding(data["crop"])

            is_dup, max_sim = self.is_duplicate(emb, kept_embeddings)

            if is_dup:
                removed[tid] = {
                    "conf": data["conf"],
                    "frame_id": data["frame_id"],
                    "max_similarity": round(max_sim, 4),
                    "similar_to": kept_ids[
                        int(np.argmax([self.cosine_similarity(emb, e) for e in kept_embeddings]))
                    ],
                }
            else:
                filtered[tid] = data
                filtered[tid]["embedding"] = emb
                kept_embeddings.append(emb)
                kept_ids.append(tid)

        n_orig = len(best_shots)
        n_kept = len(filtered)
        n_removed = len(removed)
        print(f"[Deduplicator] {n_orig} -> {n_kept} shots (removed {n_removed} duplicates)")

        return filtered, removed

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        if self._owns_classifier and hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class FinalDeduplicator:
    """
    후처리 ID 병합기.

    영상 처리 완료 후, 시간(frame_id)과 임베딩 유사도 기준으로
    서로 다른 track_id가 실제로는 동일 개체인지 판단하여 병합한다.
    """

    def __init__(
        self,
        frame_gap_threshold: int = 150,
        similarity_threshold: float = 0.80,
    ) -> None:
        self.frame_gap = frame_gap_threshold
        self.sim_thresh = similarity_threshold
        print(f"[FinalDedup] Initialized (frame_gap={self.frame_gap}, sim_thresh={self.sim_thresh})")

    def merge_overlapping_ids(self, saved_dict: dict) -> dict:
        """
        저장된 best-shot 딕셔너리에서 동일 개체 ID를 병합한다.

        병합 조건:
            |frame_id 차이| < frame_gap_threshold
            AND cosine_similarity(emb_A, emb_B) > similarity_threshold

        Args:
            saved_dict: {tid: {"conf": float, "frame_id": int, "path": str, "embedding": np.ndarray}}

        Returns:
            병합 후 딕셔너리 (중복 제거, conf 높은 쪽 유지, 파일 삭제)
        """
        if len(saved_dict) <= 1:
            return saved_dict

        ids = sorted(saved_dict.keys())

        # Union-Find
        parent: dict[int, int] = {tid: tid for tid in ids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # 비교 및 병합
        merge_count = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                data_a, data_b = saved_dict[a], saved_dict[b]

                # 임베딩 없으면 스킵
                emb_a = data_a.get("embedding")
                emb_b = data_b.get("embedding")
                if emb_a is None or emb_b is None:
                    continue

                # 조건 1: frame 간격
                frame_diff = abs(data_a["frame_id"] - data_b["frame_id"])
                if frame_diff > self.frame_gap:
                    continue

                # 조건 2: 임베딩 유사도
                sim = float(np.dot(emb_a, emb_b))
                if sim >= self.sim_thresh:
                    union(a, b)
                    merge_count += 1

        # 그룹별 대표 선택 (conf 가장 높은 것)
        groups: dict[int, list[int]] = defaultdict(list)
        for tid in ids:
            groups[find(tid)].append(tid)

        merged: dict = {}
        removed_paths: list[str] = []

        for root, members in groups.items():
            # conf 기준 정렬
            members.sort(key=lambda t: saved_dict[t]["conf"], reverse=True)
            best_tid = members[0]
            merged[best_tid] = saved_dict[best_tid]

            # 나머지 파일 삭제
            for dup_tid in members[1:]:
                path = saved_dict[dup_tid].get("path", "")
                if path and Path(path).exists():
                    Path(path).unlink()
                    removed_paths.append(path)

        n_before = len(saved_dict)
        n_after = len(merged)
        print(f"[FinalDedup] {n_before} -> {n_after} shots (merged {merge_count} pairs, deleted {len(removed_paths)} files)")

        return merged
