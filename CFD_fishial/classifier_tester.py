"""
Divery Vision Pipeline - Module B: BioCLIP-2 Classifier Tester
===============================================================
BioCLIP-2 (TreeOfLife-200M) 기반 어종 분류기 독립 테스트 모듈.
"""

import csv
import json
import os
import sys

# Suppress Hugging Face symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import collections
import heapq
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from huggingface_hub import hf_hub_download

try:
    from open_clip import create_model, get_tokenizer
except ImportError:
    print("❌ open_clip_torch is not installed. pip install open_clip_torch")
    sys.exit(1)


class ClassifierTester:
    """
    BioCLIP-2 임베딩 분류기 클래스.
    """

    def __init__(self, config_path: str = "config.yaml", model_choice: str = "bioclip2") -> None:
        self.config = self._load_config(config_path)
        cls_cfg = self.config.get("classifier", {})

        self.model_choice = model_choice
        print(f"[ClassifierTester] Loading BioCLIP model ({model_choice})...")
        self.device = torch.device(cls_cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
        
        if model_choice == "bioclip-2.5-vith14":
            self.model_str = "hf-hub:imageomics/bioclip-2.5-vith14"
            self.tokenizer_str = "hf-hub:imageomics/bioclip-2.5-vith14"
        else:
            self.model_str = cls_cfg.get("model_str", "hf-hub:imageomics/bioclip-2")
            self.tokenizer_str = cls_cfg.get("tokenizer_str", "ViT-L-14")
            
        self.hf_data_str = cls_cfg.get("hf_data_str", "imageomics/TreeOfLife-200M")

        self.model = create_model(self.model_str, output_dict=True, require_pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        self.tokenizer = get_tokenizer(self.tokenizer_str)
        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
            lambda c: f"a photo of my {c}.",
            lambda c: f"a close-up photo of a {c}.",
            lambda c: f"a bright photo of a {c}.",
            lambda c: f"a dark photo of a {c}.",
            lambda c: f"a photo of a large {c}.",
            lambda c: f"a photo of a small {c}.",
        ]

        if self.model_choice != "bioclip-2.5-vith14":
            print("[ClassifierTester] Downloading/Loading TreeOfLife-200M embeddings (768-dim)...")
            self.txt_emb = torch.from_numpy(np.load(hf_hub_download(
                repo_id=self.hf_data_str,
                filename="embeddings/txt_emb_species.npy",
                repo_type="dataset",
            ))).to(self.device)

            with open(hf_hub_download(
                repo_id=self.hf_data_str,
                filename="embeddings/txt_emb_species.json",
                repo_type="dataset",
            ), encoding="utf-8") as fd:
                self.txt_names = json.load(fd)
        else:
            self.txt_emb = None
            self.txt_names = None

        # Warmup for stable GPU timing
        print("[ClassifierTester] GPU warmup...")
        self._warmup()

        self.topk = cls_cfg.get("topk_results", 3)
        self.output_dir = Path(self.config.get("output", {}).get("output_dir", "./results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Two-stage classification setup
        self.bioinfo_file = Path("fish_bioinfo/taxon_data_filtered_keyword_matched.json")
        self.family_to_species = collections.defaultdict(list)
        self.scientific_to_common = {}
        self.family_names = []
        self.family_emb = None
        self._load_bioinfo()

        print(f"[ClassifierTester] Initialized")
        print(f"  - Device   : {self.device}")
        print(f"  - Top-K    : {self.topk}")
        if self.family_names:
            print(f"  - Families : {len(self.family_names)}")

    def _load_bioinfo(self):
        if not self.bioinfo_file.exists():
            print(f"[WARNING] Bioinfo JSON not found at {self.bioinfo_file}. Two-stage mode will be unavailable.")
            return
            
        print(f"[ClassifierTester] Loading Fish Bioinfo {self.bioinfo_file}...")
        with open(self.bioinfo_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for item in data:
            family = item.get("Family")
            family_kr = item.get("FamilyKR", "")
            species_full = item.get("SpcScitfNm")
            comm_kor_nm = item.get("CommKorNm", "")
            
            if family and species_full:
                # Extract binomial name (first two words) to match BioCLIP format
                parts = species_full.split()
                if len(parts) >= 2:
                    species = f"{parts[0]} {parts[1]}"
                else:
                    species = species_full

                # Store the family with both english and korean if available
                family_label = f"{family} ({family_kr})" if family_kr else family
                
                # Append species
                self.family_to_species[family_label].append(species)
                
                # Store korean common name mapping
                if comm_kor_nm:
                    self.scientific_to_common[species] = comm_kor_nm
        
        # Sort and extract unique families
        self.family_names = sorted(list(self.family_to_species.keys()))
        
        # Precompute embeddings for families
        if self.family_names:
            print("[ClassifierTester] Precomputing Family text embeddings...")
            self.family_emb = self.get_txt_features(self.family_names)

    def _warmup(self, num_iterations=3):
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            for _ in range(num_iterations):
                self.model.encode_image(dummy_input)

    @staticmethod
    def _load_config(config_path: str) -> dict:
        cfg = Path(config_path)
        if not cfg.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(cfg, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def format_name(self, taxon, common):
        taxon_str = " ".join(taxon)
        
        # Lookup Korean common name
        kor_nm = self.scientific_to_common.get(taxon_str, "")
        
        # Build base name
        base_name = taxon_str
        if common:
            base_name = f"{taxon_str} ({common})"
            
        # Append Korean name if it exists
        if kor_nm:
            return f"{base_name} [{kor_nm}]"
        return base_name

    @torch.no_grad()
    def get_txt_features(self, classnames):
        all_features = []
        for classname in classnames:
            txts = [template(classname) for template in self.templates]
            txts = self.tokenizer(txts).to(self.device)
            txt_features = self.model.encode_text(txts)
            txt_features = F.normalize(txt_features, dim=-1).mean(dim=0)
            txt_features /= txt_features.norm()
            all_features.append(txt_features)
        return torch.stack(all_features, dim=1)

    def run(self, image_dir: str, mode: str = "open-domain", custom_classes: Optional[list[str]] = None, target_files: Optional[list[str]] = None) -> list[dict]:
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"[WARNING] Image dir not found: {image_dir}")
            return []

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in extensions)
        
        if target_files is not None:
            image_files = [f for f in image_files if f.name in target_files]

        if not image_files:
            print(f"[WARNING] No images found in: {image_dir}")
            return []

        print(f"\n{'='*60}")
        print(f"[Classification] BioCLIP-2 ({mode})")
        print(f"  Images: {len(image_files)} | Top-K: {self.topk}")
        
        target_txt_emb = self.txt_emb
        target_names = self.txt_names
        
        if mode == "open-domain" and self.model_choice == "bioclip-2.5-vith14":
            print("[ERROR] BioCLIP-2.5 (ViT-H-14) does not currently support Open-Domain classification because the 1024-dim TreeOfLife-200M embeddings are not yet available. Please use Zero-Shot or Two-Stage classification.")
            raise ValueError("BioCLIP-2.5 does not currently support Open-Domain classification.")
            
        if mode == "zero-shot":
            if not custom_classes:
                print("[WARNING] Zero-shot mode requires custom_classes. Falling back to open-domain.")
                mode = "open-domain"
            else:
                print(f"  Classes: {len(custom_classes)} custom species")
                target_txt_emb = self.get_txt_features(custom_classes)
                # Store names as tuple similar to JSON format (e.g. ['Amphiprion ocellaris'], '')
                target_names = [([cls], "") for cls in custom_classes]
        elif mode == "zero-shot-all":
            if not self.scientific_to_common:
                print("[WARNING] Bioinfo JSON not loaded properly. Falling back to open-domain.")
                mode = "open-domain"
            else:
                custom_classes = list(self.scientific_to_common.keys())
                print(f"  Classes: {len(custom_classes)} species from Bioinfo JSON")
                target_txt_emb = self.get_txt_features(custom_classes)
                target_names = [([cls], "") for cls in custom_classes]
        elif mode == "two-stage":
            if not self.family_names or self.family_emb is None:
                print("[WARNING] Bioinfo JSON not loaded properly. Falling back to open-domain.")
                mode = "open-domain"
            else:
                print("  Mode   : Two-stage (Family -> Species)")

        print(f"{'='*60}")

        results: list[dict] = []

        with tqdm(image_files, desc="[Classify]", unit="img") as pbar:
            for img_path in pbar:
                if mode == "two-stage":
                    result = self._classify_two_stage(img_path)
                else:
                    result = self._classify_single(img_path, target_txt_emb, target_names)
                    
                if result is not None:
                    results.append(result)
                    pbar.set_postfix(
                        top1=result.get("Top1_Species", "?")[:15],
                        conf=f"{result.get('Top1_Prob', 0):.3f}",
                    )

        csv_name = self.config.get("classifier", {}).get("csv_filename", "classification_log.csv")
        csv_path = self.output_dir / csv_name
        self._save_csv(results, csv_path)
        self._print_summary(results)
        return results

    @torch.no_grad()
    def _classify_single(self, img_path: Path, target_txt_emb: torch.Tensor, target_names: list) -> Optional[dict]:
        try:
            img_pil = Image.open(img_path).convert("RGB")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            img_tensor = self.preprocess(img_pil).to(self.device)
            img_features = self.model.encode_image(img_tensor.unsqueeze(0))
            img_features = F.normalize(img_features, dim=-1)

            logits = (self.model.logit_scale.exp() * img_features @ target_txt_emb).squeeze()
            probs = F.softmax(logits, dim=0)

            # Prevent error if topk exceeds number of classes
            k = min(self.topk, len(probs))
            topk_res = probs.topk(k)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            inference_ms = (t_end - t_start) * 1000

            row: dict = {"Image_Name": img_path.name}
            
            if k == 1 and logits.dim() == 0:
                # Edge case for a single class
                row[f"Top1_Species"] = self.format_name(*target_names[0])
                row[f"Top1_Prob"] = round(probs.item(), 4)
            else:
                for k_idx, (idx, prob) in enumerate(zip(topk_res.indices, topk_res.values)):
                    rank = k_idx + 1
                    species_name = self.format_name(*target_names[idx.item()])
                    row[f"Top{rank}_Species"] = species_name
                    row[f"Top{rank}_Prob"] = round(prob.item(), 4)

            # Fill in the rest with empty
            for k_idx in range(k, self.topk):
                rank = k_idx + 1
                row[f"Top{rank}_Species"] = ""
                row[f"Top{rank}_Prob"] = 0.0

            row["Inference_Time(ms)"] = round(inference_ms, 2)
            return row

        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            return None

    @torch.no_grad()
    def _classify_two_stage(self, img_path: Path) -> Optional[dict]:
        try:
            img_pil = Image.open(img_path).convert("RGB")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            # 1. Encode Image
            img_tensor = self.preprocess(img_pil).to(self.device)
            img_features = self.model.encode_image(img_tensor.unsqueeze(0))
            img_features = F.normalize(img_features, dim=-1)

            # 2. Predict Family
            logits_fam = (self.model.logit_scale.exp() * img_features @ self.family_emb).squeeze()
            probs_fam = F.softmax(logits_fam, dim=0)
            
            top1_fam_idx = probs_fam.argmax().item()
            pred_family = self.family_names[top1_fam_idx]
            
            # 3. Retrieve Candidate Species based on predicted Family
            candidate_species = list(set(self.family_to_species[pred_family]))
            
            # 4. Predict Species (Zero-shot)
            if not candidate_species:
                # Fallback if somehow no species
                candidate_emb = self.txt_emb
                candidate_names = self.txt_names
            else:
                candidate_emb = self.get_txt_features(candidate_species)
                candidate_names = [([cls], "") for cls in candidate_species]
                
            logits_spc = (self.model.logit_scale.exp() * img_features @ candidate_emb).squeeze()
            probs_spc = F.softmax(logits_spc, dim=0)
            
            k = min(self.topk, len(probs_spc))
            topk_res = probs_spc.topk(k)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            inference_ms = (t_end - t_start) * 1000

            row: dict = {"Image_Name": img_path.name}
            row["Pred_Family"] = pred_family
            
            if k == 1 and logits_spc.dim() == 0:
                row[f"Top1_Species"] = self.format_name(*candidate_names[0])
                row[f"Top1_Prob"] = round(probs_spc.item(), 4)
            else:
                for k_idx, (idx, prob) in enumerate(zip(topk_res.indices, topk_res.values)):
                    rank = k_idx + 1
                    species_name = self.format_name(*candidate_names[idx.item()])
                    row[f"Top{rank}_Species"] = species_name
                    row[f"Top{rank}_Prob"] = round(prob.item(), 4)

            for k_idx in range(k, self.topk):
                rank = k_idx + 1
                row[f"Top{rank}_Species"] = ""
                row[f"Top{rank}_Prob"] = 0.0

            row["Inference_Time(ms)"] = round(inference_ms, 2)
            return row

        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            return None

    def _save_csv(self, results: list[dict], csv_path: Path) -> None:
        if not results:
            return
        fieldnames = ["Image_Name"]
        if "Pred_Family" in results[0]:
            fieldnames.append("Pred_Family")
        for k in range(1, self.topk + 1):
            fieldnames += [f"Top{k}_Species", f"Top{k}_Prob"]
        fieldnames.append("Inference_Time(ms)")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"[Save] Classification log -> {csv_path}")

    @staticmethod
    def _print_summary(results: list[dict]) -> None:
        if not results:
            return
        n = len(results)
        probs = [r.get("Top1_Prob", 0) for r in results]
        times = [r.get("Inference_Time(ms)", 0) for r in results]
        avg_prob = sum(probs) / n
        avg_time = sum(times) / n
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

    def cleanup(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[ClassifierTester] GPU memory released.")
