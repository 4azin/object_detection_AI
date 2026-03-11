"""
Divery Vision Pipeline - Integrated Web UI
===========================================
Gradio 기반 통합 웹 인터페이스.
Step 1: 영상 업로드 -> ByteTrack 추적 -> Best-Shot 추출
Step 2: Best-Shot 갤러리 확인
Step 3: BioCLIP-2 어종 분류

사용법:
    conda activate diveary-vision
    cd CFD_fishial
    python web_app.py
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
import yaml

from best_shot_extractor import BestShotExtractor
from detector_tester import DetectorTester

# ──────────────────────────────────────────────
# Globals (lazy init)
# ──────────────────────────────────────────────
_extractor: Optional[BestShotExtractor] = None
_classifier = None  # ClassifierTester (lazy import to avoid heavy init)
_last_best_shots_dir: Optional[str] = None


def _get_extractor() -> BestShotExtractor:
    global _extractor
    if _extractor is None:
        _extractor = BestShotExtractor(config_path="config.yaml")
    return _extractor


def _get_classifier():
    global _classifier
    if _classifier is None:
        from classifier_tester import ClassifierTester
        _classifier = ClassifierTester(config_path="config.yaml")
    return _classifier


# ──────────────────────────────────────────────
# Step 1: Video -> Best-Shot Extraction
# ──────────────────────────────────────────────
def step1_extract(
    video_file: str,
    conf_threshold: float,
    use_fp16: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """Video upload -> ByteTrack -> Best-Shot extraction + rendered tracking video."""
    global _last_best_shots_dir

    if video_file is None:
        raise gr.Error("Please upload a video file.")

    extractor = _get_extractor()

    # Override config with UI values
    extractor.config["tracker"]["confidence_threshold"] = conf_threshold
    extractor.config["model"]["half"] = use_fp16

    # Use persistent results directory (timestamped subfolder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = Path(video_file).stem
    run_dir = Path("results") / "best_shots" / f"{video_stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    extractor.best_shots_dir = run_dir
    extractor.output_dir = Path("results")

    _last_best_shots_dir = str(extractor.best_shots_dir)

    # --- Run tracking with rendered video ---
    video_path = Path(video_file)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Video writer for rendered output
    out_video_path = run_dir / f"tracked_{video_path.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps_orig, (width, height))

    tracker_cfg = extractor.config["tracker"]
    model_cfg = extractor.config["model"]
    best_shots: dict = {}
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

    from tqdm import tqdm
    with tqdm(total=total_frames, desc="[Track]", unit="frame") as pbar:
        for result in results:
            frame = result.orig_img
            boxes = result.boxes

            # Render bounding boxes + IDs on frame
            rendered = result.plot(img=frame.copy())
            writer.write(rendered)

            # Best-shot logic
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

    # --- Safeguard: Dedup + Merge ---
    from deduplicator import EmbeddingDeduplicator, FinalDeduplicator

    safeguard = extractor.config.get("safeguard", {})
    emb_thresh = safeguard.get("embedding_similarity_threshold", 0.85)
    merge_gap = safeguard.get("merge_frame_gap", 150)
    merge_sim = safeguard.get("merge_similarity_threshold", 0.80)

    dedup = EmbeddingDeduplicator(
        config_path="config.yaml", similarity_threshold=emb_thresh
    )
    filtered, removed = dedup.deduplicate_best_shots(best_shots)

    # Save filtered best-shots
    saved = extractor._save_best_shots(filtered, video_path.stem, fps_orig)

    # Final ID merge
    final_dedup = FinalDeduplicator(
        frame_gap_threshold=merge_gap, similarity_threshold=merge_sim
    )
    merged = final_dedup.merge_overlapping_ids(saved)

    dedup.cleanup()

    _last_best_shots_dir = str(extractor.best_shots_dir)

    # Build gallery images from merged results
    gallery_items = []
    for tid in sorted(merged.keys()):
        data = merged[tid]
        img_path = data["path"]
        img = cv2.imread(img_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = f"ID {tid} | Conf: {data['conf']:.3f} | Frame: {data['frame_id']}"
            gallery_items.append((img_rgb, label))

    # Summary HTML (with dedup stats)
    n_raw = len(best_shots)
    n_final = len(merged)
    summary = _build_extraction_summary(merged, video_file, n_raw, len(removed))
    gpu_html = _build_gpu_info_html()
    tracked_video = str(out_video_path) if out_video_path.exists() else None

    # Create ZIP for download
    zip_path = _create_best_shot_zip(extractor.best_shots_dir)

    return gallery_items, summary, gpu_html, str(extractor.best_shots_dir), tracked_video, zip_path


def step1_extract_image(
    image,
    conf_threshold: float,
    use_fp16: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """Image upload -> YOLO detect -> Best-Shot extraction."""
    global _last_best_shots_dir

    if image is None:
        raise gr.Error("Please upload an image.")

    extractor = _get_extractor()

    # Override config with UI values
    extractor.config["tracker"]["confidence_threshold"] = conf_threshold
    extractor.config["model"]["half"] = use_fp16

    # Use persistent results directory (timestamped subfolder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / "best_shots" / f"image_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    extractor.best_shots_dir = run_dir
    extractor.output_dir = Path("results")

    _last_best_shots_dir = str(extractor.best_shots_dir)

    # Convert PIL Image to BGR for OpenCV
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

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
    
    # Assign dummy track IDs
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
    rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)

    # --- Safeguard: Dedup (skip merge since it's a single frame) ---
    from deduplicator import EmbeddingDeduplicator
    safeguard = extractor.config.get("safeguard", {})
    emb_thresh = safeguard.get("embedding_similarity_threshold", 0.85)

    dedup = EmbeddingDeduplicator(
        config_path="config.yaml", similarity_threshold=emb_thresh
    )
    filtered, removed = dedup.deduplicate_best_shots(best_shots)
    saved = extractor._save_best_shots(filtered, "image", 30.0)
    dedup.cleanup()

    _last_best_shots_dir = str(extractor.best_shots_dir)

    gallery_items = []
    for tid in sorted(saved.keys()):
        data = saved[tid]
        img_path = data["path"]
        crop_img = cv2.imread(img_path)
        if crop_img is not None:
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            label = f"Obj {tid} | Conf: {data['conf']:.3f}"
            gallery_items.append((crop_rgb, label))

    # Summary HTML (with dedup stats)
    n_raw = len(best_shots)
    n_final = len(saved)
    summary = _build_extraction_summary(saved, "image_upload", n_raw, len(removed))
    gpu_html = _build_gpu_info_html()
    zip_path = _create_best_shot_zip(extractor.best_shots_dir)

    return gallery_items, summary, gpu_html, str(extractor.best_shots_dir), rendered_rgb, zip_path


def _create_best_shot_zip(best_shots_dir: Path) -> Optional[str]:
    """Best-shot 폴더를 ZIP으로 압축하여 다운로드 경로를 반환한다."""
    best_shots_dir = Path(best_shots_dir)
    jpg_files = list(best_shots_dir.glob("*.jpg"))
    if not jpg_files:
        return None
    zip_base = best_shots_dir.parent / best_shots_dir.name
    zip_path = shutil.make_archive(str(zip_base), "zip", str(best_shots_dir))
    return zip_path


def step1_download_zip(best_shots_path: str):
    """기존 best-shots 폴더로부터 ZIP을 재생성하여 다운로드 경로를 반환한다."""
    if not best_shots_path or not Path(best_shots_path).exists():
        raise gr.Error("No best-shots found. Please run Step 1 first.")
    zip_path = _create_best_shot_zip(Path(best_shots_path))
    if zip_path is None:
        raise gr.Error("No .jpg files found in best-shots directory.")
    return zip_path


def _build_extraction_summary(saved: dict, video_path: str, n_raw: int = 0, n_dedup_removed: int = 0) -> str:
    if not saved:
        return "<p>No tracked objects found.</p>"

    n = len(saved)
    confs = [d["conf"] for d in saved.values()]
    avg_conf = sum(confs) / n
    max_conf = max(confs)
    min_conf = min(confs)

    dedup_html = ""
    if n_raw > 0:
        n_merge_removed = n_raw - n_dedup_removed - n
        dedup_html = f"""
        <div style="margin-top:12px; padding:12px; background:rgba(255,255,255,0.05);
            border-radius:12px; font-size:13px;">
            <b style="color:#a29bfe;">Safeguard Pipeline</b><br>
            <span style="color:#aaa;">Raw IDs: {n_raw}</span>
            <span style="color:#ff6b6b; margin-left:8px;">Embedding dedup: -{n_dedup_removed}</span>
            <span style="color:#ffd93d; margin-left:8px;">ID merge: -{n_merge_removed}</span>
            <span style="color:#6bff6b; margin-left:8px;">Final: {n}</span>
        </div>
        """

    return f"""
    <div style="
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px; padding: 24px; color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    ">
        <h3 style="margin:0 0 16px 0; color:#00d2ff;
            border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom:12px;">
            Step 1 Complete - {Path(video_path).name}
        </h3>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px;">
            <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:32px; font-weight:700; color:#00d2ff;">{n}</div>
                <div style="font-size:12px; color:#aaa; margin-top:4px;">Final Unique</div>
            </div>
            <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:32px; font-weight:700; color:#6bff6b;">{avg_conf:.3f}</div>
                <div style="font-size:12px; color:#aaa; margin-top:4px;">Avg Confidence</div>
            </div>
            <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:32px; font-weight:700; color:#ffd93d;">{max_conf:.3f}</div>
                <div style="font-size:12px; color:#aaa; margin-top:4px;">Max Confidence</div>
            </div>
            <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:32px; font-weight:700; color:#ff9f43;">{min_conf:.3f}</div>
                <div style="font-size:12px; color:#aaa; margin-top:4px;">Min Confidence</div>
            </div>
        </div>
        {dedup_html}
        <p style="margin-top:12px; color:#aaa; font-size:13px;">
            Best shots saved. Proceed to Step 2 to classify species.
        </p>
    </div>
    """


# ──────────────────────────────────────────────
# Step 2: Classify Best-Shots
# ──────────────────────────────────────────────
def step2_classify(
    best_shots_path: str,
    classification_mode: str,
    zero_shot_classes: str,
    progress=gr.Progress(track_tqdm=True),
):
    """Classify all best-shot images with BioCLIP-2."""
    if not best_shots_path or not Path(best_shots_path).exists():
        raise gr.Error("No best-shots found. Please run Step 1 first.")

    mode = "open-domain" if "Open-Domain" in classification_mode else "zero-shot"
    custom_classes = []
    if mode == "zero-shot":
        if not zero_shot_classes.strip():
            raise gr.Error("Please provide candidate species for Zero-Shot mode.")
        custom_classes = [c.strip() for c in zero_shot_classes.split("\n") if c.strip()]

    classifier = _get_classifier()
    results = classifier.run(best_shots_path, mode=mode, custom_classes=custom_classes)

    if not results:
        return None, "<p>No classification results.</p>", []

    # Build DataFrame
    df = pd.DataFrame(results)

    # Build comparison gallery: each image with its Top-3 species annotation
    comparison_items = []
    for row in results:
        img_path = Path(best_shots_path) / row["Image_Name"]
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Annotate with Top-3
                annotated = _annotate_classification(img_rgb, row)
                label = (
                    f"{row['Image_Name']}\n"
                    f"1st: {row['Top1_Species']} ({row['Top1_Prob']:.3f})\n"
                    f"2nd: {row['Top2_Species']} ({row['Top2_Prob']:.3f})\n"
                    f"3rd: {row['Top3_Species']} ({row['Top3_Prob']:.3f})"
                )
                comparison_items.append((annotated, label))

    summary_html = _build_classification_summary(results)
    gpu_html = _build_gpu_info_html()

    return df, summary_html, comparison_items, gpu_html


def _annotate_classification(img_rgb: np.ndarray, row: dict) -> np.ndarray:
    """Draw Top-3 species labels on the image."""
    h, w = img_rgb.shape[:2]
    # Scale for readability
    scale = max(1.0, min(w, h) / 200)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(scale))
    font_scale = 0.45 * scale

    # Convert to BGR for OpenCV drawing, then back
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Draw semi-transparent overlay at bottom
    overlay_h = int(70 * scale)
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, h - overlay_h), (w, h), (0, 0, 0), -1)
    img_bgr = cv2.addWeighted(overlay, 0.7, img_bgr, 0.3, 0)

    colors = [(0, 255, 200), (0, 200, 255), (200, 200, 200)]  # BGR: cyan, yellow, gray
    y_start = h - overlay_h + int(18 * scale)

    for k in range(3):
        rank = k + 1
        species = row.get(f"Top{rank}_Species", "")
        prob = row.get(f"Top{rank}_Prob", 0.0)
        if species:
            text = f"{rank}. {species} ({prob:.2f})"
            y_pos = y_start + int(k * 20 * scale)
            cv2.putText(img_bgr, text, (int(5 * scale), y_pos),
                        font, font_scale, colors[k], thickness, cv2.LINE_AA)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _build_classification_summary(results: list[dict]) -> str:
    if not results:
        return ""

    n = len(results)
    probs = [r.get("Top1_Prob", 0) for r in results]
    times = [r.get("Inference_Time(ms)", 0) for r in results]
    avg_prob = sum(probs) / n
    avg_time = sum(times) / n

    from collections import Counter
    species_counts = Counter(r.get("Top1_Species", "") for r in results)
    top_species = species_counts.most_common(5)

    species_rows = ""
    for species, count in top_species:
        pct = count / n * 100
        bar_w = pct  # directly use percentage as width
        species_rows += f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <span style="width:180px; text-align:right; font-size:13px; color:#ddd;
                overflow:hidden; text-overflow:ellipsis; white-space:nowrap;"
                title="{species}">{species}</span>
            <div style="flex:1; background:rgba(255,255,255,0.1); border-radius:6px; height:20px;">
                <div style="width:{bar_w}%; height:100%; border-radius:6px;
                    background: linear-gradient(90deg, #00d2ff, #928DAB);
                    display:flex; align-items:center; justify-content:flex-end;
                    padding-right:6px; font-size:11px; color:#000; font-weight:600;
                    min-width:40px;">
                    {count} ({pct:.0f}%)
                </div>
            </div>
        </div>
        """

    return f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        border-radius: 16px; padding: 24px; color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    ">
        <h3 style="margin:0 0 16px 0; color:#00d2ff;
            border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom:12px;">
            Step 2 Complete - Species Classification
        </h3>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:16px;">
            <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:28px; font-weight:700; color:#00d2ff;">{n}</div>
                <div style="font-size:12px; color:#aaa; margin-top:4px;">Images Classified</div>
            </div>
            <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:28px; font-weight:700; color:#6bff6b;">{avg_prob:.4f}</div>
                <div style="font-size:12px; color:#aaa; margin-top:4px;">Avg Top-1 Confidence</div>
            </div>
            <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:28px; font-weight:700; color:#ff9f43;">{avg_time:.1f} ms</div>
                <div style="font-size:12px; color:#aaa; margin-top:4px;">Avg Inference Time</div>
            </div>
        </div>
        <h4 style="margin:0 0 8px 0; color:#a29bfe;">Top-5 Predicted Species</h4>
        {species_rows}
    </div>
    """


# ──────────────────────────────────────────────
# GPU info helper
# ──────────────────────────────────────────────
def _build_gpu_info_html() -> str:
    if not torch.cuda.is_available():
        return '<p style="color:#ff6b6b;">GPU not available.</p>'

    gpu_name = torch.cuda.get_device_name(0)
    mem_alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
    mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    pct = (mem_alloc / mem_total) * 100 if mem_total > 0 else 0
    bar_color = "#6bff6b" if pct < 60 else "#ffd93d" if pct < 85 else "#ff6b6b"

    return f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px; padding: 16px; color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif; font-size: 13px;
    ">
        <b style="color:#00d2ff;">GPU:</b> {gpu_name}<br>
        <div style="background:rgba(255,255,255,0.1); border-radius:6px; height:18px;
            margin-top:6px; overflow:hidden;">
            <div style="background:linear-gradient(90deg,{bar_color},{bar_color}88);
                width:{pct:.0f}%; height:100%; border-radius:6px;
                display:flex; align-items:center; justify-content:center;
                font-size:10px; font-weight:600; color:#000; min-width:55px;">
                {mem_alloc:.1f}/{mem_total:.1f} GB
            </div>
        </div>
    </div>
    """


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────
CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
    font-family: 'Segoe UI', 'Inter', sans-serif !important;
}
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 28px 20px;
    margin-bottom: 8px;
    color: white;
}
.main-header h1 {
    margin: 0; font-size: 30px;
    background: linear-gradient(90deg, #00d2ff, #928DAB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-header p { margin: 6px 0 0 0; color: #aaa; font-size: 13px; }
"""


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="Divery Pipeline") as demo:
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>Divery Vision Pipeline</h1>
            <p>CFD Detector (YOLOv12x) + ByteTrack + BioCLIP-2 (TreeOfLife-200M)</p>
        </div>
        """)

        # Hidden state for best_shots directory path
        best_shots_state = gr.State(value="")

        with gr.Tabs():
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Tab 1: Video -> Best-Shot Extraction
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Step 1: Tracking & Best-Shot"):
                with gr.Tabs():
                    with gr.Tab("Video Tracking"):
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=1):
                                gr.Markdown("### Upload Video")
                                video_input = gr.Video(
                                    label="Dive Video",
                                    sources=["upload"],
                                )
                                with gr.Accordion("Model Settings", open=True):
                                    conf_slider = gr.Slider(
                                        0.1, 0.9, value=0.5, step=0.05,
                                        label="Tracking Confidence",
                                        info="ByteTrack conf threshold",
                                    )
                                    fp16_check = gr.Checkbox(
                                        value=True, label="FP16 (RTX 4070)",
                                    )
                                extract_btn = gr.Button(
                                    "Start Tracking & Extraction",
                                    variant="primary", size="lg",
                                )
                                download_btn = gr.Button(
                                    "📦 Download All Best Shots (ZIP)",
                                    variant="secondary", size="lg",
                                )
                                zip_download = gr.File(
                                    label="Download ZIP",
                                    visible=True,
                                )
                                gpu_info_1 = gr.HTML(value=_build_gpu_info_html())

                            with gr.Column(scale=2):
                                extract_summary = gr.HTML(label="Extraction Summary")
                                with gr.Tabs():
                                    with gr.Tab("Tracked Video"):
                                        tracked_video_output = gr.Video(
                                            label="Tracking Result (Bounding Boxes + IDs)",
                                        )
                                    with gr.Tab("Best-Shot Gallery"):
                                        extract_gallery = gr.Gallery(
                                            label="Extracted Best Shots (per tracked ID)",
                                            columns=4,
                                            height=460,
                                            object_fit="contain",
                                        )

                        extract_btn.click(
                            fn=step1_extract,
                            inputs=[video_input, conf_slider, fp16_check],
                            outputs=[extract_gallery, extract_summary, gpu_info_1, best_shots_state, tracked_video_output, zip_download],
                        )

                        download_btn.click(
                            fn=step1_download_zip,
                            inputs=[best_shots_state],
                            outputs=[zip_download],
                        )
                    
                    with gr.Tab("Image Detection"):
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=1):
                                gr.Markdown("### Upload Image")
                                image_input = gr.Image(
                                    type="pil",
                                    label="Dive Image",
                                    sources=["upload"],
                                )
                                with gr.Accordion("Model Settings", open=True):
                                    img_conf_slider = gr.Slider(
                                        0.1, 0.9, value=0.5, step=0.05,
                                        label="Detection Confidence",
                                        info="YOLO conf threshold",
                                    )
                                    img_fp16_check = gr.Checkbox(
                                        value=True, label="FP16 (RTX 4070)",
                                    )
                                img_extract_btn = gr.Button(
                                    "Start Detection & Extraction",
                                    variant="primary", size="lg",
                                )
                                img_download_btn = gr.Button(
                                    "📦 Download All Objects (ZIP)",
                                    variant="secondary", size="lg",
                                )
                                img_zip_download = gr.File(
                                    label="Download ZIP",
                                    visible=True,
                                )
                                img_gpu_info_1 = gr.HTML(value=_build_gpu_info_html())

                            with gr.Column(scale=2):
                                img_extract_summary = gr.HTML(label="Extraction Summary")
                                with gr.Tabs():
                                    with gr.Tab("Detected Image"):
                                        detected_image_output = gr.Image(
                                            label="Detection Result (Bounding Boxes)",
                                        )
                                    with gr.Tab("Object Gallery"):
                                        img_extract_gallery = gr.Gallery(
                                            label="Extracted Objects",
                                            columns=4,
                                            height=460,
                                            object_fit="contain",
                                        )

                        img_extract_btn.click(
                            fn=step1_extract_image,
                            inputs=[image_input, img_conf_slider, img_fp16_check],
                            outputs=[img_extract_gallery, img_extract_summary, img_gpu_info_1, best_shots_state, detected_image_output, img_zip_download],
                        )

                        img_download_btn.click(
                            fn=step1_download_zip,
                            inputs=[best_shots_state],
                            outputs=[img_zip_download],
                        )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Tab 2: BioCLIP-2 Classification
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Step 2: Species Classification"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.Markdown("### Classify Best-Shots")
                        
                        classify_mode = gr.Radio(
                            choices=["🌊 Open-Domain (TreeOfLife-200M)", "🎯 Zero-Shot (Custom List)"],
                            value="🌊 Open-Domain (TreeOfLife-200M)",
                            label="Classification Mode",
                        )
                        zs_classes = gr.Textbox(
                            label="Zero-Shot Candidates",
                            placeholder="Amphiprion ocellaris (Clownfish)\nParacanthurus hepatus (Blue Tang)\nZebrasoma flavescens (Yellow Tang)",
                            lines=5,
                            info="Required for Zero-Shot. Separated by newlines.",
                            visible=False,
                        )

                        def toggle_zs_input(mode):
                            return gr.update(visible="Zero-Shot" in mode)
                        classify_mode.change(fn=toggle_zs_input, inputs=classify_mode, outputs=zs_classes)

                        classify_btn = gr.Button(
                            "Start Classification",
                            variant="primary", size="lg",
                        )
                        gpu_info_2 = gr.HTML(value=_build_gpu_info_html())

                    with gr.Column(scale=2):
                        classify_summary = gr.HTML(label="Classification Summary")
                        gr.Markdown("### Species Comparison (Top-3 annotated)")
                        classify_gallery = gr.Gallery(
                            label="Classification Results",
                            columns=3,
                            height=520,
                            object_fit="contain",
                        )

                with gr.Accordion("Detailed Classification Log (CSV)", open=False):
                    classify_table = gr.Dataframe(
                        label="classification_log",
                        interactive=False,
                        wrap=True,
                    )

                classify_btn.click(
                    fn=step2_classify,
                    inputs=[best_shots_state, classify_mode, zs_classes],
                    outputs=[classify_table, classify_summary, classify_gallery, gpu_info_2],
                )

    return demo


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        max_file_size="500mb",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
    )
