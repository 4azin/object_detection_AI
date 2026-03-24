"""
Divery Vision Pipeline - CLI Integration (GPU Server Version)
=============================================================
웹프론트엔드(Gradio) 없이 백그라운드나 GPU 서버에서 단일 스크립트로 동작하도록
web_app.py의 로직(Step 1: Extract, Step 2: Classify)을 병합한 통합 코드입니다.

사용법:
    conda activate diveary-vision
    cd CFD_fishial
    python integration.py --source <path_to_video_or_image> --mode open-domain
"""

import argparse
import sys
import traceback
from pathlib import Path
from datetime import datetime

print("[DEBUG] Starting imports...")

try:
    import cv2
    print("[DEBUG] cv2 imported")
except Exception as e:
    print(f"[ERROR] cv2 import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    import pandas as pd
    print("[DEBUG] pandas imported")
except Exception as e:
    print(f"[ERROR] pandas import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    import numpy as np
    print("[DEBUG] numpy imported")
except Exception as e:
    print(f"[ERROR] numpy import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    import torch
    print(f"[DEBUG] torch imported, CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"[ERROR] torch import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from tqdm import tqdm
    print("[DEBUG] tqdm imported")
except Exception as e:
    print(f"[ERROR] tqdm import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from best_shot_extractor import BestShotExtractor
    print("[DEBUG] BestShotExtractor imported")
except Exception as e:
    print(f"[ERROR] BestShotExtractor import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from deduplicator import EmbeddingDeduplicator, FinalDeduplicator
    print("[DEBUG] deduplicator imported")
except Exception as e:
    print(f"[ERROR] deduplicator import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from classifier_tester import ClassifierTester
    print("[DEBUG] ClassifierTester imported")
except Exception as e:
    print(f"[ERROR] ClassifierTester import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from background_remover import BackgroundRemover, REMBG_AVAILABLE
    print(f"[DEBUG] background_remover imported, REMBG_AVAILABLE: {REMBG_AVAILABLE}")
except Exception as e:
    print(f"[ERROR] background_remover import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("[DEBUG] All imports successful!")

def get_device_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return f"Using GPU: {gpu_name}"
    return "GPU not available, using CPU (not recommended)."

def run_extraction_video(video_path, conf_threshold, use_fp16, extractor):
    print(f"\n--- [Step 1] Video Best-Shot Extraction ---")
    print(f"Tracking config: conf={conf_threshold}, fp16={use_fp16}")

    extractor.config["tracker"]["confidence_threshold"] = conf_threshold
    extractor.config["model"]["half"] = use_fp16

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = Path(video_path).stem
    run_dir = Path("results") / "best_shots" / f"{video_stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    extractor.best_shots_dir = run_dir
    extractor.output_dir = Path("results")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_video_path = run_dir / f"tracked_{video_stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps_orig, (width, height))

    tracker_cfg = extractor.config["tracker"]
    model_cfg = extractor.config["model"]
    best_shots = {}
    frame_id = 0

    results = extractor.model.track(
        source=str(video_path),
        persist=True,
        tracker=tracker_cfg["tracker_type"],
        conf=tracker_cfg["confidence_threshold"],
        iou=model_cfg["iou_threshold"],
        imgsz=model_cfg["img_size"],
        half=model_cfg["half"],
        device=extractor.device,
        stream=True,
        verbose=False,
    )

    with tqdm(total=total_frames, desc="[Track]", unit="frame") as pbar:
        for result in results:
            frame = result.orig_img
            boxes = result.boxes

            rendered = result.plot(img=frame.copy())
            writer.write(rendered)

            if boxes is not None and boxes.id is not None:
                for i in range(len(boxes)):
                    tid = int(boxes.id[i].item())
                    conf = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)

                    if tid not in best_shots or conf > best_shots[tid]["conf"]:
                        x1, y1, x2, y2 = xyxy
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
            pbar.set_postfix(tracked=len(best_shots))
            pbar.update(1)
            frame_id += 1

    writer.release()

    print("\n--- [Safeguard] Deduplication ---")
    safeguard = extractor.config.get("safeguard", {})
    emb_thresh = safeguard.get("embedding_similarity_threshold", 0.85)
    merge_gap = safeguard.get("merge_frame_gap", 150)
    merge_sim = safeguard.get("merge_similarity_threshold", 0.80)

    dedup = EmbeddingDeduplicator(config_path="config.yaml", similarity_threshold=emb_thresh)
    filtered, removed = dedup.deduplicate_best_shots(best_shots)
    saved = extractor._save_best_shots(filtered, video_stem, fps_orig)

    final_dedup = FinalDeduplicator(frame_gap_threshold=merge_gap, similarity_threshold=merge_sim)
    merged = final_dedup.merge_overlapping_ids(saved)
    dedup.cleanup()

    print(f"Tracking complete. Raw IDs: {len(best_shots)}, After Embedding Dedup: {len(filtered)}, Final Merged: {len(merged)}")
    print(f"Tracking video saved to: {out_video_path}")
    return run_dir, merged


def run_extraction_image(image_path, conf_threshold, use_fp16, extractor):
    print(f"\n--- [Step 1] Image Best-Shot Extraction ---")
    print(f"[DEBUG] run_extraction_image: image_path={image_path}")

    try:
        extractor.config["tracker"]["confidence_threshold"] = conf_threshold
        extractor.config["model"]["half"] = use_fp16
        print(f"[DEBUG] config updated: conf={conf_threshold}, fp16={use_fp16}")
    except Exception as e:
        print(f"[ERROR] config update failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_stem = Path(image_path).stem
    run_dir = Path("results") / "best_shots" / f"image_{image_stem}_{timestamp}"
    print(f"[DEBUG] run_dir={run_dir}")

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] run_dir created")
    except Exception as e:
        print(f"[ERROR] run_dir creation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    extractor.best_shots_dir = run_dir
    extractor.output_dir = Path("results")

    print(f"[DEBUG] reading image: {image_path}")
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Error: Could not read image {image_path}")
        sys.exit(1)
    print(f"[DEBUG] image read successfully, shape={img_bgr.shape}")

    results = extractor.model.predict(
        source=img_bgr,
        conf=conf_threshold,
        iou=extractor.config["model"]["iou_threshold"],
        imgsz=extractor.config["model"]["img_size"],
        half=use_fp16,
        device=extractor.device,
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes
    best_shots = {}

    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            tid = i
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)

            x1, y1, x2, y2 = xyxy
            h, w = img_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crop = img_bgr[y1:y2, x1:x2].copy()
                best_shots[tid] = {
                    "conf": conf,
                    "crop": crop,
                    "frame_id": 0,
                }

    rendered = result.plot(img=img_bgr.copy())
    out_img_path = run_dir / f"detected_{image_stem}.jpg"
    cv2.imwrite(str(out_img_path), rendered)

    print("\n--- [Safeguard] Deduplication ---")
    safeguard = extractor.config.get("safeguard", {})
    emb_thresh = safeguard.get("embedding_similarity_threshold", 0.85)

    dedup = EmbeddingDeduplicator(config_path="config.yaml", similarity_threshold=emb_thresh)
    filtered, removed = dedup.deduplicate_best_shots(best_shots)
    saved = extractor._save_best_shots(filtered, "image", 30.0)
    dedup.cleanup()

    print(f"Detection complete. Raw Objects: {len(best_shots)}, Final Saved: {len(saved)}")
    print(f"Rendered image saved to: {out_img_path}")
    return run_dir, saved


def run_classification(best_shots_dir, target_files, model_choice, mode, custom_classes):
    print(f"\n--- [Step 2] Species Classification ---")
    print(f"Model: {model_choice}, Mode: {mode}")

    if mode == "open-domain" and "2.5" in model_choice:
        print("Error: BioCLIP-2.5 does not currently support open-domain mode. Please use 'bioclip2'.")
        sys.exit(1)

    classifier = ClassifierTester(config_path="config.yaml", model_choice=model_choice)
    results = classifier.run(best_shots_dir, mode=mode, custom_classes=custom_classes, target_files=target_files)

    if not results:
        print("No classification results.")
        return None

    df = pd.DataFrame(results)

    csv_path = Path(best_shots_dir) / "classification_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nClassification completed. Results saved to {csv_path}")

    probs = [r.get("Top1_Prob", 0) for r in results]
    avg_prob = sum(probs) / len(results)
    print(f"\nClassification Summary:")
    print(f"  Images Classified: {len(results)}")
    print(f"  Avg Top-1 Confidence: {avg_prob:.4f}")

    print("\nTop Predicts:")
    for row in results[:10]:
        print(f"  - {row['Image_Name']}: {row.get('Top1_Species','')} ({row.get('Top1_Prob',0):.3f})")

    return df


def main():
    print("[DEBUG] main() started")
    parser = argparse.ArgumentParser(description="Divery Vision Pipeline - CLI Integration")
    parser.add_argument("--source", type=str, required=True, help="Path to input video or image file")
    parser.add_argument("--conf", type=float, default=0.5, help="Tracking/Detection confidence threshold (default: 0.5)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 precision (use full precision)")
    parser.add_argument("--model", type=str, choices=["bioclip2", "bioclip-2.5-vith14"], default="bioclip2", help="Classification model")
    parser.add_argument("--mode", type=str, choices=["open-domain", "zero-shot", "zero-shot-all", "two-stage"], default="open-domain", help="Classification mode")
    parser.add_argument("--classes", type=str, default="", help="Comma-separated list of candidate species for zero-shot mode")
    parser.add_argument("--remove-bg", action="store_true", help="Remove background from fish crops (requires rembg)")
    parser.add_argument("--bg-model", type=str, default="u2net", help="Background removal model (default: u2net)")

    args = parser.parse_args()
    print(f"[DEBUG] args parsed: source={args.source}, conf={args.conf}, model={args.model}, mode={args.mode}")

    use_fp16 = not args.no_fp16
    source_path = Path(args.source)
    print(f"[DEBUG] source_path={source_path}, exists={source_path.exists()}")

    if not source_path.exists():
        print(f"Error: Source file '{args.source}' does not exist.")
        sys.exit(1)

    print(get_device_info())

    print("[DEBUG] Creating BestShotExtractor...")
    try:
        extractor = BestShotExtractor(config_path="config.yaml")
        print("[DEBUG] BestShotExtractor created successfully")
    except Exception as e:
        print(f"[ERROR] BestShotExtractor creation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    ext = source_path.suffix.lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        run_dir, extracted_data = run_extraction_video(source_path, args.conf, use_fp16, extractor)
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        run_dir, extracted_data = run_extraction_image(source_path, args.conf, use_fp16, extractor)
    else:
        print(f"Error: Unsupported file extension {ext}")
        sys.exit(1)

    if not extracted_data:
        print("\nNo objects detected. Exiting classification step.")
        sys.exit(0)

    # Background removal (optional)
    if args.remove_bg:
        if REMBG_AVAILABLE:
            print(f"\n--- [Background Removal] ---")
            remover = BackgroundRemover(model_name=args.bg_model, keep_original=True)
            extracted_data = remover.process_best_shots(extracted_data)
            remover.cleanup()
        else:
            print("\n[WARNING] --remove-bg specified but rembg is not installed. Skipping.")
            print("  Install with: pip install rembg[gpu] or pip install rembg")

    target_files = [Path(d["path"]).name for d in extracted_data.values()]

    custom_classes = []
    if args.mode == "zero-shot":
        if not args.classes:
            print("Error: Zero-shot mode requires --classes to be provided.")
            sys.exit(1)
        custom_classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    run_classification(str(run_dir), target_files, args.model, args.mode, custom_classes)

    print("\n[Pipeline execution finished successfully!]")


if __name__ == "__main__":
    main()
