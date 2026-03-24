from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import io
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urlparse

import boto3
import cv2
import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("divery-ai-worker")

RUNNING = True


def _on_signal(signum: int, _frame: Any) -> None:
    global RUNNING
    log.info("received signal=%s, shutting down", signum)
    RUNNING = False


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value.strip() == ""):
        raise RuntimeError(f"missing required env: {name}")
    return value or ""


def parse_float(value: str, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def parse_bool(value: str, fallback: bool = False) -> bool:
    normalized = (value or "").strip().lower()
    if not normalized:
        return fallback
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return fallback


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


class Worker:
    def __init__(self) -> None:
        self.region = env("AWS_REGION", "ap-northeast-2")
        self.bucket = env("S3_BUCKET_NAME", required=True)
        self.cloudfront_hosts = self._parse_cloudfront_hosts(
            env("CLOUDFRONT_DOMAIN", ""),
            env("CLOUDFRONT_DOMAINS", ""),
        )
        self.queue_url = env("SQS_QUEUE_URL", required=True)
        self.callback_base_url = env("BACKEND_CALLBACK_BASE_URL", "").rstrip("/")
        self.callback_secret = env("AI_CALLBACK_SIGNING_SECRET", "")
        self.poll_wait_seconds = int(env("SQS_WAIT_TIME_SECONDS", "20"))
        self.visibility_timeout = int(env("SQS_VISIBILITY_TIMEOUT", "900"))
        self.callback_timeout = int(env("CALLBACK_TIMEOUT_SECONDS", "10"))
        self.download_timeout = int(env("DOWNLOAD_TIMEOUT_SECONDS", "120"))
        self.pipeline_timeout = int(env("PIPELINE_TIMEOUT_SECONDS", "7200"))
        self.dummy_mode = env("WORKER_DUMMY_MODE", "false").lower() in {"1", "true", "yes", "y"}
        self.keep_workdir = env("WORKER_KEEP_WORKDIR", "false").lower() in {"1", "true", "yes", "y"}
        self.callback_retry_count = max(1, int(env("CALLBACK_RETRY_COUNT", "3")))
        self.callback_retry_delay_seconds = max(1, int(env("CALLBACK_RETRY_DELAY_SECONDS", "2")))
        self.visibility_heartbeat_seconds = max(
            10,
            int(env("SQS_VISIBILITY_HEARTBEAT_SECONDS", str(max(30, self.visibility_timeout // 3)))),
        )

        worker_root = Path(__file__).resolve().parents[1]
        self.tmp_root = Path(env("WORKER_TMP_DIR", "/tmp/divery-ai-worker"))
        self.tmp_root.mkdir(parents=True, exist_ok=True)

        cfd_dir_default = str(worker_root / "CFD_fishial")
        self.cfd_dir = Path(env("CFD_PIPELINE_DIR", cfd_dir_default)).resolve()
        self.cfd_script = Path(env("CFD_INTEGRATION_SCRIPT", str(self.cfd_dir / "integration.py"))).resolve()
        self.cfd_python = env("CFD_PYTHON_BIN", sys.executable)
        self.cfd_model = env("CFD_MODEL", "bioclip2")
        self.cfd_mode = env("CFD_MODE", "open-domain")
        self.cfd_conf = env("CFD_CONF_THRESHOLD", "0.5")
        self.cfd_classes = env("CFD_ZERO_SHOT_CLASSES", "")
        self.cfd_disable_fp16 = env("CFD_DISABLE_FP16", "false").lower() in {"1", "true", "yes", "y"}
        self.cfd_remove_bg = env("CFD_REMOVE_BG", "true").lower() in {"1", "true", "yes", "y"}
        self.cfd_bg_model = env("CFD_BG_MODEL", "u2net").strip()
        self.highlight_top_k = max(1, int(env("HIGHLIGHT_TOP_K", "5")))
        self.highlight_min_gap_seconds = max(0, int(env("HIGHLIGHT_MIN_GAP_SECONDS", "2")))
        highlight_model_default = str(self.cfd_dir / "weights" / "yolov8l-world.pt")
        self.highlight_model_weights = Path(env("HIGHLIGHT_MODEL_WEIGHTS", highlight_model_default)).resolve()
        self.highlight_model_conf = parse_float(env("HIGHLIGHT_MODEL_CONF", "0.1"), 0.1)
        self.highlight_model: YOLO | None = None
        
        # Add NIMA Initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nima_weights_default = str(worker_root.parent / "weights" / "nima_resnet50_finetuned.pth")
        self.nima_weights_path = Path(env("NIMA_MODEL_WEIGHTS", nima_weights_default)).resolve()
        
        self.nima_model = NIMA_ResNet50().to(self.device)
        if self.nima_weights_path.exists():
            try:
                self.nima_model.load_state_dict(torch.load(str(self.nima_weights_path), map_location=self.device))
                log.info(f"NIMA weights loaded from {self.nima_weights_path}")
            except Exception as e:
                log.warning(f"Failed to load NIMA weights: {e}")
        else:
            log.warning(f"NIMA weights not found at {self.nima_weights_path}. Using default initialized weights.")
        self.nima_model.eval()
        
        self.nima_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.gms_key = env("GMS_KEY", "").strip()
        llm_enabled_raw = env("LOGBOOK_LLM_ENABLED", "auto").strip().lower()
        if llm_enabled_raw == "auto":
            self.logbook_llm_enabled = bool(self.gms_key)
        else:
            self.logbook_llm_enabled = parse_bool(llm_enabled_raw, False)
        self.logbook_llm_url = env(
            "LOGBOOK_LLM_URL",
            "https://gms.ssafy.io/gmsapi/api.openai.com/v1/chat/completions",
        ).strip()
        self.logbook_llm_model = env("LOGBOOK_LLM_MODEL", "gpt-5-mini").strip()
        self.logbook_llm_timeout = max(5, int(env("LOGBOOK_LLM_TIMEOUT_SECONDS", "20")))
        self.logbook_llm_retries = max(1, int(env("LOGBOOK_LLM_RETRY_COUNT", "2")))
        self.logbook_llm_temperature = parse_float(env("LOGBOOK_LLM_TEMPERATURE", "0.4"), 0.4)
        self.logbook_llm_max_tokens = max(64, int(env("LOGBOOK_LLM_MAX_TOKENS", "300")))
        self.logbook_llm_max_chars = max(120, int(env("LOGBOOK_LLM_MAX_CHARS", "320")))

        # Vision LLM settings for image analysis
        self.vision_llm_enabled = parse_bool(env("VISION_LLM_ENABLED", "true"), True) and bool(self.gms_key)
        self.vision_llm_model = env("VISION_LLM_MODEL", "gpt-5-mini").strip()
        self.synthesis_llm_model = env("SYNTHESIS_LLM_MODEL", "o3-mini").strip()
        self.vision_llm_timeout = max(10, int(env("VISION_LLM_TIMEOUT_SECONDS", "30")))
        self.anthropic_url = env(
            "ANTHROPIC_LLM_URL",
            "https://gms.ssafy.io/gmsapi/api.anthropic.com/v1/messages",
        ).strip()
        self.vision_image_max_size = (512, 512)

        self.sqs = boto3.client("sqs", region_name=self.region)
        self.s3 = boto3.client("s3", region_name=self.region)
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}"

        log.info(
            "worker initialized region=%s queue=%s bucket=%s dummy_mode=%s worker_id=%s",
            self.region,
            self.queue_url,
            self.bucket,
            self.dummy_mode,
            self.worker_id,
        )
        if self.cloudfront_hosts:
            log.info("cloudfront fallback hosts=%s", ",".join(sorted(self.cloudfront_hosts)))
        if not self.dummy_mode:
            log.info(
                "pipeline config cfd_dir=%s cfd_script=%s model=%s mode=%s conf=%s highlight_top_k=%s",
                self.cfd_dir,
                self.cfd_script,
                self.cfd_model,
                self.cfd_mode,
                self.cfd_conf,
                self.highlight_top_k,
            )
        if self.logbook_llm_enabled and not self.gms_key:
            self.logbook_llm_enabled = False
            log.warning("logbook llm disabled: GMS_KEY is empty")
        if self.logbook_llm_enabled:
            log.info("logbook llm enabled model=%s url=%s", self.logbook_llm_model, self.logbook_llm_url)
        if self.vision_llm_enabled:
            log.info("vision llm enabled vision_model=%s synthesis_model=%s", self.vision_llm_model, self.synthesis_llm_model)

    def validate_preflight(self) -> None:
        """Validate all required resources before starting the worker loop.

        Checks:
        1. AWS credentials (STS get-caller-identity)
        2. S3 bucket access
        3. SQS queue access
        4. CFD integration script existence
        5. Model weight files existence

        Raises:
            RuntimeError: If any validation check fails.
        """
        log.info("running preflight validation...")
        errors: List[str] = []

        # 1. AWS credentials validation via STS
        try:
            sts = boto3.client("sts", region_name=self.region)
            identity = sts.get_caller_identity()
            log.info("preflight: AWS credentials valid account=%s arn=%s", identity.get("Account"), identity.get("Arn"))
        except Exception as exc:
            errors.append(f"AWS credentials invalid: {exc}")

        # 2. S3 bucket access validation
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            log.info("preflight: S3 bucket accessible bucket=%s", self.bucket)
        except Exception as exc:
            errors.append(f"S3 bucket not accessible ({self.bucket}): {exc}")

        # 3. SQS queue access validation
        try:
            self.sqs.get_queue_attributes(QueueUrl=self.queue_url, AttributeNames=["QueueArn"])
            log.info("preflight: SQS queue accessible queue=%s", self.queue_url)
        except Exception as exc:
            errors.append(f"SQS queue not accessible ({self.queue_url}): {exc}")

        # 4. CFD integration script existence (skip in dummy mode)
        if not self.dummy_mode:
            if not self.cfd_script.exists():
                errors.append(f"CFD integration script not found: {self.cfd_script}")
            else:
                log.info("preflight: CFD script found path=%s", self.cfd_script)

        # 5. Model weight files existence (skip in dummy mode)
        if not self.dummy_mode:
            required_weights = [
                self.highlight_model_weights,
            ]
            # Check for CFD model weights
            cfd_model_path = self.cfd_dir / "cfd-yolov12x-1.00.pt"
            if cfd_model_path.exists():
                log.info("preflight: CFD model weights found path=%s", cfd_model_path)
            else:
                # Also check for any .pt files in cfd_dir
                pt_files = list(self.cfd_dir.glob("*.pt"))
                if pt_files:
                    log.info("preflight: Found model weights in cfd_dir: %s", [f.name for f in pt_files])
                else:
                    errors.append(f"No model weight files found in CFD directory: {self.cfd_dir}")

            # Check highlight model weights
            if self.highlight_model_weights.exists():
                log.info("preflight: Highlight model weights found path=%s", self.highlight_model_weights)
            else:
                log.warning("preflight: Highlight model weights not found path=%s (will skip highlight scoring)", self.highlight_model_weights)

        if errors:
            error_msg = "Preflight validation failed:\n  - " + "\n  - ".join(errors)
            log.error(error_msg)
            raise RuntimeError(error_msg)

        log.info("preflight validation passed")

    def run(self) -> None:
        self.validate_preflight()
        while RUNNING:
            response = self.sqs.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=self.poll_wait_seconds,
                VisibilityTimeout=self.visibility_timeout,
                MessageAttributeNames=["All"],
                AttributeNames=["All"],
            )
            messages = response.get("Messages", [])
            if not messages:
                continue

            for message in messages:
                self._handle_message(message)

    def _handle_message(self, message: Dict[str, Any]) -> None:
        receipt_handle = message["ReceiptHandle"]
        message_id = message.get("MessageId", "")
        started_at = time.time()
        stop_visibility = threading.Event()
        visibility_thread = threading.Thread(
            target=self._visibility_heartbeat_loop,
            args=(receipt_handle, stop_visibility),
            daemon=True,
        )
        visibility_thread.start()

        try:
            payload = json.loads(message["Body"])
            task_id = payload["task_id"]
            persona = payload.get("persona") or "기록형"
            upload_session_id = self._extract_upload_session_id(payload)
            callback = payload.get("callback") or {}

            self._callback_processing(task_id, callback, upload_session_id)

            result_payload = self._process(task_id, persona, payload)
            self._callback_complete(task_id, result_payload, callback)

            self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)
            elapsed_ms = int((time.time() - started_at) * 1000)
            log.info("message done message_id=%s task_id=%s elapsed_ms=%s", message_id, task_id, elapsed_ms)
        except Exception as exc:  # noqa: BLE001
            task_id = "unknown"
            try:
                parsed = json.loads(message.get("Body", "{}"))
                task_id = parsed.get("task_id", "unknown")
                upload_session_id = self._extract_upload_session_id(parsed)
                callback = parsed.get("callback") or {}
                self._callback_fail(
                    task_id,
                    "WORKER_PROCESSING_ERROR",
                    str(exc),
                    callback,
                    upload_session_id,
                )
            except Exception:  # noqa: BLE001
                log.error("failed to send fail callback task_id=%s", task_id)

            log.error("message failed task_id=%s error=%s", task_id, exc)
            log.error(traceback.format_exc())
            # Failure path keeps message unacked so retry/DLQ handles it.
        finally:
            stop_visibility.set()
            visibility_thread.join(timeout=3)

    def _process(self, task_id: str, persona: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        upload_session_id = self._extract_upload_session_id(payload)
        if self.dummy_mode:
            result_payload = self._dummy_result(task_id, persona, upload_session_id)
            self._upload_result_to_s3(task_id, result_payload)
            return result_payload

        work_dir = Path(tempfile.mkdtemp(prefix=f"{task_id}-", dir=str(self.tmp_root)))
        log.info("task=%s work_dir=%s", task_id, work_dir)

        try:
            frame_keys = self._extract_frame_keys(payload)
            if frame_keys:
                frames = self._download_frames(frame_keys, work_dir)
                selected_highlights = self._select_highlights_from_frames(frames, self.highlight_top_k)
                cfd_result = self._run_cfd_for_highlights(selected_highlights, work_dir)
            else:
                video_path = self._download_video(payload, work_dir)
                cfd_result = self._run_cfd_integration(video_path, work_dir)

            species = cfd_result.get("species", [])
            detections = cfd_result.get("detections", [])
            highlights = cfd_result.get("highlights", [])
            logbook = self._build_logbook(persona, species, detections, highlights)

            result_payload = {
                "taskId": task_id,
                "status": "completed",
                "persona": persona,
                "logbook": logbook,
                "highlights": highlights,
                "detections": detections,
                "species": species,
                "completedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            if upload_session_id:
                result_payload["uploadSessionId"] = upload_session_id
            self._upload_result_to_s3(task_id, result_payload)
            return result_payload
        finally:
            if self.keep_workdir:
                log.info("task=%s keep work_dir=%s", task_id, work_dir)
            else:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _dummy_result(self, task_id: str, persona: str, upload_session_id: str | None = None) -> Dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload = {
            "taskId": task_id,
            "status": "completed",
            "persona": persona,
            "logbook": f"[{persona}] queue worker processed task {task_id}",
            "highlights": [],
            "detections": [],
            "species": [],
            "completedAt": now_iso,
        }
        if upload_session_id:
            payload["uploadSessionId"] = upload_session_id
        return payload

    @staticmethod
    def _extract_upload_session_id(payload: Dict[str, Any]) -> str | None:
        value = str(payload.get("upload_session_id") or payload.get("uploadSessionId") or "").strip()
        return value or None

    def _extract_frame_keys(self, payload: Dict[str, Any]) -> List[str]:
        raw = payload.get("frame_keys")
        if not isinstance(raw, list):
            return []

        keys: List[str] = []
        seen: Set[str] = set()
        for item in raw:
            key = str(item or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            keys.append(key)
        return keys

    def _download_frames(self, frame_keys: List[str], work_dir: Path) -> List[Dict[str, Any]]:
        frames_dir = work_dir / "input_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_items: List[Dict[str, Any]] = []
        for index, frame_key in enumerate(frame_keys):
            parsed = urlparse(frame_key)
            suffix = Path(parsed.path).suffix.lower() or ".jpg"
            local_path = frames_dir / f"frame_{index:05d}{suffix}"
            self._download_media_reference(frame_key, local_path)
            frame_items.append(
                {
                    "index": index,
                    "frameKey": frame_key,
                    "second": self._extract_second_from_key(frame_key, index),
                    "localPath": str(local_path),
                }
            )

        if not frame_items:
            raise RuntimeError("no downloaded frame from frame_keys")
        return frame_items

    def _download_media_reference(self, media_ref: str, local_path: Path) -> None:
        ref = str(media_ref or "").strip()
        if not ref:
            raise RuntimeError("media reference is empty")

        parsed = urlparse(ref)
        if parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            if not bucket or not key:
                raise RuntimeError(f"invalid s3 media reference: {ref}")
            self.s3.download_file(bucket, key, str(local_path))
            return

        if parsed.scheme in {"http", "https"}:
            if self._try_http_download(ref, local_path):
                return
            bucket, key = self._extract_s3_location_from_http_url(ref)
            if bucket and key:
                self.s3.download_file(bucket, key, str(local_path))
                return
            raise RuntimeError(f"failed to download media via http and could not parse s3 url: {ref}")

        # Treat non-url values as S3 object keys in worker default bucket.
        self.s3.download_file(self.bucket, ref.lstrip("/"), str(local_path))

    @staticmethod
    def _extract_second_from_key(frame_key: str, fallback: int) -> int:
        name = Path(urlparse(frame_key).path).name
        match = re.search(r"(?:frame[_-]?)(\d+)", name, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except Exception:  # noqa: BLE001
                return fallback
        match = re.search(r"(\d+)", name)
        if match:
            try:
                return int(match.group(1))
            except Exception:  # noqa: BLE001
                return fallback
        return fallback

    def _select_highlights_from_frames(self, frame_items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not frame_items:
            return []

        scored: List[Dict[str, Any]] = []
        for item in frame_items:
            score, detection_count = self._score_highlight_frame(Path(item["localPath"]))
            scored.append(
                {
                    **item,
                    "highlightScore": round(score, 4),
                    "detectionCount": detection_count,
                }
            )

        ordered = sorted(scored, key=lambda row: (float(row["highlightScore"]), -int(row["index"])), reverse=True)
        selected: List[Dict[str, Any]] = []
        for candidate in ordered:
            if len(selected) >= top_k:
                break
            if self.highlight_min_gap_seconds > 0 and selected:
                too_close = any(
                    abs(int(candidate["second"]) - int(prev["second"])) < self.highlight_min_gap_seconds
                    for prev in selected
                )
                if too_close:
                    continue
            selected.append(candidate)

        if len(selected) < min(top_k, len(ordered)):
            selected_ids = {item["index"] for item in selected}
            for candidate in ordered:
                if candidate["index"] in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate["index"])
                if len(selected) >= min(top_k, len(ordered)):
                    break

        selected.sort(key=lambda row: int(row["index"]))
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank

        return selected

    def _score_highlight_frame(self, image_path: Path) -> Tuple[float, int]:
        image = cv2.imread(str(image_path))
        if image is None:
            return 0.0, 0

        # NIMA Score
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            nima_tensor = self.nima_transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                nima_prob = self.nima_model(nima_tensor)
                scores_tensor = torch.arange(1, 11).float().to(self.device)
                mean_score = torch.sum(nima_prob * scores_tensor, dim=1).item()
                n_score = (mean_score - 1) / 9.0
        except Exception as exc:
            log.warning("NIMA scoring failed image=%s error=%s", image_path, exc)
            n_score = 0.0

        # YOLO Score (Object Detection)
        detection_count = 0
        y_score = 0.0
        
        if self.highlight_model is not None or self.highlight_model_weights.exists():
            try:
                model = self._ensure_highlight_model()
                results = model.predict(
                    source=image,
                    conf=self.highlight_model_conf,
                    imgsz=640,
                    verbose=False,
                )
                if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    confs = boxes.conf.cpu().tolist()
                    xyxy = boxes.xyxy.cpu().numpy()
                    frame_area = float(max(1, image.shape[0] * image.shape[1]))
                    area_ratio_sum = 0.0
                    for box in xyxy:
                        x1, y1, x2, y2 = box
                        area_ratio_sum += max(0.0, float((x2 - x1) * (y2 - y1)) / frame_area)

                    detection_count = len(confs)
                    max_conf = max(confs) if confs else 0.0
                    area_score = min(area_ratio_sum, 1.0)
                    count_bonus = min(detection_count / 10.0, 0.2)
                    y_score = (0.65 * max_conf) + (0.25 * area_score) + count_bonus
            except Exception as exc:
                log.warning("Highlight yolov8 scoring fallback image=%s error=%s", image_path, exc)

        # Fallback sharpness score if YOLO completely fails
        if y_score == 0.0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            y_score = min(sharpness / 400.0, 0.2)

        # Final Combined Score (0.6 YOLO + 0.4 NIMA)
        final_score = (0.6 * y_score) + (0.4 * n_score)
        return final_score, detection_count

    def _ensure_highlight_model(self) -> YOLO:
        if self.highlight_model is not None:
            return self.highlight_model
        if not self.highlight_model_weights.exists():
            raise RuntimeError(f"highlight model weights not found: {self.highlight_model_weights}")
        self.highlight_model = YOLO(str(self.highlight_model_weights))
        return self.highlight_model

    def _run_cfd_for_highlights(self, highlights: List[Dict[str, Any]], work_dir: Path) -> Dict[str, Any]:
        if not highlights:
            return {"species": [], "detections": [], "highlights": []}

        all_detections: List[Dict[str, Any]] = []
        merged_species: Dict[str, Dict[str, Any]] = {}
        enriched_highlights: List[Dict[str, Any]] = []

        for item in highlights:
            image_path = Path(item["localPath"])
            cfd_result = self._run_cfd_integration(image_path, work_dir)

            for detection in cfd_result.get("detections", []):
                row = dict(detection)
                row["sourceFrameKey"] = item.get("frameKey")
                row["sourceSecond"] = item.get("second")
                row["highlightRank"] = item.get("rank")
                all_detections.append(row)

            for species_item in cfd_result.get("species", []):
                key = str(
                    species_item.get("scientificName")
                    or species_item.get("name")
                    or species_item.get("commonName")
                    or species_item.get("koreanName")
                    or ""
                ).lower()
                if not key:
                    continue
                existing = merged_species.get(key)
                if existing is None:
                    merged_species[key] = dict(species_item)
                else:
                    existing["count"] = int(existing.get("count", 0)) + int(species_item.get("count", 0))
                    existing["confidence"] = round(
                        max(float(existing.get("confidence", 0.0)), float(species_item.get("confidence", 0.0))),
                        4,
                    )

            top_species = None
            if cfd_result.get("species"):
                top_species = cfd_result["species"][0].get("name")

            enriched_highlights.append(
                {
                    "rank": item.get("rank"),
                    "second": item.get("second"),
                    "frameKey": item.get("frameKey"),
                    "localPath": item.get("localPath"),
                    "highlightScore": item.get("highlightScore"),
                    "detectionCount": len(cfd_result.get("detections", [])),
                    "topSpecies": top_species,
                }
            )

        merged_species_list = sorted(
            merged_species.values(),
            key=lambda item: (-float(item.get("confidence", 0.0)), -int(item.get("count", 0)), str(item.get("name"))),
        )
        for idx, item in enumerate(merged_species_list, start=1):
            item["speciesId"] = idx

        return {
            "species": merged_species_list,
            "detections": all_detections,
            "highlights": enriched_highlights,
        }

    def _download_video(self, payload: Dict[str, Any], work_dir: Path) -> Path:
        video_url = str(payload.get("video_url") or "").strip()
        if not video_url:
            raise RuntimeError("queue payload missing video_url")

        parsed = urlparse(video_url)
        suffix = Path(parsed.path).suffix.lower() or ".mp4"
        local_video = work_dir / f"input{suffix}"

        if parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            if not bucket or not key:
                raise RuntimeError(f"invalid s3 video_url: {video_url}")
            self.s3.download_file(bucket, key, str(local_video))
            return local_video

        if parsed.scheme in {"http", "https"}:
            if self._try_http_download(video_url, local_video):
                return local_video

            bucket, key = self._extract_s3_location_from_http_url(video_url)
            if bucket and key:
                self.s3.download_file(bucket, key, str(local_video))
                return local_video

            raise RuntimeError(f"failed to download video via http and could not parse s3 url: {video_url}")

        raise RuntimeError(f"unsupported video_url scheme: {video_url}")

    def _try_http_download(self, video_url: str, local_video: Path) -> bool:
        try:
            with requests.get(video_url, stream=True, timeout=self.download_timeout) as response:
                if response.status_code >= 400:
                    log.warning("http video download failed status=%s url=%s", response.status_code, video_url)
                    return False

                with local_video.open("wb") as fp:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fp.write(chunk)
                return True
        except Exception as exc:  # noqa: BLE001
            log.warning("http video download exception url=%s error=%s", video_url, exc)
            return False

    def _extract_s3_location_from_http_url(self, url: str) -> Tuple[str | None, str | None]:
        parsed = urlparse(url)
        host = self._normalize_host(parsed.netloc or "")
        path = parsed.path.lstrip("/")

        # virtual-hosted-style: <bucket>.s3.<region>.amazonaws.com/<key>
        m1 = re.match(r"^(?P<bucket>.+?)\.s3(?:[.-][^.]+)?\.amazonaws\.com$", host)
        if m1 and path:
            return m1.group("bucket"), path

        # path-style: s3.<region>.amazonaws.com/<bucket>/<key>
        m2 = re.match(r"^s3(?:[.-][^.]+)?\.amazonaws\.com$", host)
        if m2 and "/" in path:
            bucket, key = path.split("/", 1)
            return bucket, key

        # CloudFront/custom CDN host: use worker bucket with object key path.
        # Backend can return CloudFront URLs for private objects that require S3 fallback.
        if path and self._is_cloudfront_like_host(host):
            return self.bucket, path

        return None, None

    @staticmethod
    def _normalize_host(raw: str) -> str:
        value = (raw or "").strip()
        if not value:
            return ""
        parsed = urlparse(value if "://" in value else f"https://{value}")
        host = (parsed.netloc or parsed.path).strip().lower()
        if ":" in host:
            host = host.split(":", 1)[0]
        return host

    def _parse_cloudfront_hosts(self, *values: str) -> Set[str]:
        hosts: Set[str] = set()
        for value in values:
            if not value:
                continue
            for token in value.split(","):
                host = self._normalize_host(token)
                if host:
                    hosts.add(host)
        return hosts

    def _is_cloudfront_like_host(self, host: str) -> bool:
        normalized = self._normalize_host(host)
        if not normalized:
            return False
        if normalized.endswith(".cloudfront.net"):
            return True
        return normalized in self.cloudfront_hosts

    def _run_cfd_integration(self, video_path: Path, work_dir: Path) -> Dict[str, Any]:
        if not self.cfd_script.exists():
            raise RuntimeError(f"CFD integration script not found: {self.cfd_script}")

        cmd = [
            self.cfd_python,
            str(self.cfd_script),
            "--source",
            str(video_path),
            "--model",
            self.cfd_model,
            "--mode",
            self.cfd_mode,
            "--conf",
            self.cfd_conf,
        ]
        if self.cfd_disable_fp16:
            cmd.append("--no-fp16")
        if self.cfd_mode == "zero-shot" and self.cfd_classes:
            cmd.extend(["--classes", self.cfd_classes])
        if self.cfd_remove_bg:
            cmd.append("--remove-bg")
            cmd.extend(["--bg-model", self.cfd_bg_model])

        start_ts = time.time()
        log.info("running cfd pipeline cmd=%s", " ".join(cmd))

        completed = subprocess.run(
            cmd,
            cwd=str(self.cfd_dir),
            capture_output=True,
            text=True,
            timeout=self.pipeline_timeout,
            env=os.environ.copy(),
            check=False,
        )

        stdout_file = work_dir / "cfd_stdout.log"
        stderr_file = work_dir / "cfd_stderr.log"
        stdout_file.write_text(completed.stdout or "", encoding="utf-8")
        stderr_file.write_text(completed.stderr or "", encoding="utf-8")

        if completed.returncode != 0:
            raise RuntimeError(
                f"CFD pipeline failed rc={completed.returncode} stderr={self._tail(completed.stderr, 1200)}"
            )

        run_dir = self._find_latest_run_dir(start_ts)
        csv_path = run_dir / "classification_results.csv"
        if not csv_path.exists():
            raise RuntimeError(f"classification_results.csv not found in run_dir={run_dir}")

        species, detections = self._parse_classification_csv(csv_path)
        highlights = self._collect_highlights(run_dir)

        return {
            "species": species,
            "detections": detections,
            "highlights": highlights,
            "run_dir": str(run_dir),
            "csv_path": str(csv_path),
        }

    def _find_latest_run_dir(self, start_ts: float) -> Path:
        base = self.cfd_dir / "results" / "best_shots"
        if not base.exists():
            raise RuntimeError(f"CFD results directory not found: {base}")

        candidates = [
            d for d in base.iterdir()
            if d.is_dir() and d.stat().st_mtime >= start_ts - 5
        ]
        if not candidates:
            candidates = [d for d in base.iterdir() if d.is_dir()]

        if not candidates:
            raise RuntimeError(f"no run directory found under: {base}")

        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _parse_classification_csv(self, csv_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        detections: List[Dict[str, Any]] = []
        species_map: Dict[str, Dict[str, Any]] = {}

        with csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                top1_raw = (row.get("Top1_Species") or "").strip()
                top1_prob = parse_float(row.get("Top1_Prob") or "0")
                image_name = (row.get("Image_Name") or "").strip()

                if not top1_raw:
                    continue

                scientific, common_name, korean_name = self._split_species_label(top1_raw)
                display_name = korean_name or common_name or scientific
                species_name = display_name or top1_raw

                detections.append(
                    {
                        "imageName": image_name,
                        "speciesName": species_name,
                        "scientificName": scientific or None,
                        "commonName": common_name or None,
                        "koreanName": korean_name or None,
                        "confidence": round(top1_prob, 4),
                    }
                )

                key = (scientific or top1_raw).lower()
                current = species_map.get(key)
                if current is None:
                    species_map[key] = {
                        "speciesId": len(species_map) + 1,
                        "name": species_name,
                        "scientificName": scientific or None,
                        "commonName": common_name or None,
                        "koreanName": korean_name or None,
                        "confidence": round(top1_prob, 4),
                        "count": 1,
                    }
                else:
                    current["count"] = int(current.get("count", 0)) + 1
                    if top1_prob > float(current.get("confidence", 0.0)):
                        current["confidence"] = round(top1_prob, 4)

        species = sorted(species_map.values(), key=lambda item: (-float(item.get("confidence", 0.0)), item["name"]))
        return species, detections

    def _split_species_label(self, raw: str) -> Tuple[str, str, str]:
        # examples:
        # - "Acanthopagrus schlegelii (Blackhead seabream) [감성돔]"
        # - "Acanthopagrus schlegelii (Blackhead seabream)"
        # - "Acanthopagrus schlegelii"
        text = raw.strip()

        korean_name = ""
        m_kor = re.search(r"\[([^\]]+)\]\s*$", text)
        if m_kor:
            korean_name = m_kor.group(1).strip()
            text = text[:m_kor.start()].strip()

        common_name = ""
        scientific = text
        m_common = re.search(r"^(.*?)\(([^)]+)\)\s*$", text)
        if m_common:
            scientific = m_common.group(1).strip()
            common_name = m_common.group(2).strip()

        return scientific, common_name, korean_name

    def _collect_highlights(self, run_dir: Path) -> List[Dict[str, Any]]:
        highlights: List[Dict[str, Any]] = []
        image_ext = {".jpg", ".jpeg", ".png", ".webp"}

        for image_path in sorted(run_dir.iterdir()):
            if image_path.suffix.lower() not in image_ext:
                continue
            highlights.append(
                {
                    "fileName": image_path.name,
                    "localPath": str(image_path),
                }
            )
            if len(highlights) >= self.highlight_top_k:
                break

        return highlights

    def _build_logbook(
        self,
        persona: str,
        species: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
        highlights: List[Dict[str, Any]] | None = None,
    ) -> str:
        fallback = self._build_logbook_fallback(persona, species, detections, highlights)

        # Priority 1: Vision LLM pipeline (analyzes actual images)
        if self.vision_llm_enabled and highlights:
            try:
                vision_logbook = self._build_logbook_with_vision_llm(persona, highlights, species)
                if vision_logbook:
                    log.info("logbook generated via vision llm pipeline len=%d", len(vision_logbook))
                    return vision_logbook
            except Exception as exc:  # noqa: BLE001
                log.warning("vision llm pipeline failed, falling back: %s", exc)

        # Priority 2: Text-based LLM from detection data
        if not self.logbook_llm_enabled:
            return fallback
        if not detections:
            return fallback
        try:
            generated = self._build_logbook_with_llm(persona, species, detections, highlights)
            if generated:
                return generated
        except Exception as exc:  # noqa: BLE001
            log.warning("logbook llm generation failed task_persona=%s error=%s", persona, exc)
        return fallback

    def _build_logbook_fallback(
        self,
        persona: str,
        species: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
        highlights: List[Dict[str, Any]] | None = None,
    ) -> str:
        analyzed_frames = len(highlights or [])
        if not detections:
            if analyzed_frames > 0:
                return f"[{persona}] 하이라이트 {analyzed_frames}장을 분석했지만 감지된 생물 객체가 없습니다."
            return f"[{persona}] 영상 분석 결과, 감지된 생물 객체가 없습니다."

        top_names = [item.get("name") for item in species[:5] if item.get("name")]
        if not top_names:
            top_names = [item.get("speciesName") for item in detections[:5] if item.get("speciesName")]

        summary = ", ".join(str(name) for name in top_names if name)
        if analyzed_frames > 0:
            seconds = sorted(
                {
                    int(item["second"])
                    for item in (highlights or [])
                    if isinstance(item, dict) and item.get("second") is not None
                }
            )
            time_part = ""
            if seconds:
                preview = ", ".join(str(sec) for sec in seconds[:5])
                time_part = f" (하이라이트 초: {preview})"
            return f"[{persona}] 하이라이트 {analyzed_frames}장에서 주요 생물 {summary} 등이 관찰되었습니다.{time_part}"
        return f"[{persona}] 영상에서 주요 생물 {summary} 등이 관찰되었습니다."

    def _build_logbook_with_llm(
        self,
        persona: str,
        species: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
        highlights: List[Dict[str, Any]] | None = None,
    ) -> str:
        persona_prompt_map = {
            "기록형": (
                "당신은 전문 수중 기록관이다. 사실 중심으로 간결한 다이빙 로그를 작성한다. "
                "확신하지 못하는 내용은 추측하지 않는다."
            ),
            "예술가형": (
                "당신은 예술적 표현을 쓰는 다이버다. 과장 없이 감각적인 문장으로 다이빙 장면을 요약한다."
            ),
            "실속형": (
                "당신은 실용적인 다이빙 가이드다. 핵심 정보 위주로 간결하게 요약한다."
            ),
            "블로거형": (
                "당신은 SNS 스타일의 다이버다. 밝고 경쾌한 톤으로 쓰되 사실에서 벗어나지 않는다."
            ),
        }

        persona_hint = persona_prompt_map.get(persona, persona_prompt_map["기록형"])
        highlights = highlights or []
        seconds = sorted(
            {
                int(item["second"])
                for item in highlights
                if isinstance(item, dict) and item.get("second") is not None
            }
        )
        species_summary: List[Dict[str, Any]] = []
        for item in species[:8]:
            species_summary.append(
                {
                    "name": item.get("name") or item.get("speciesName"),
                    "confidence": round(float(item.get("confidence", 0.0)), 4),
                    "count": int(item.get("count", 0)),
                }
            )

        detection_summary: List[Dict[str, Any]] = []
        for item in detections[:10]:
            detection_summary.append(
                {
                    "speciesName": item.get("speciesName"),
                    "confidence": round(float(item.get("confidence", 0.0)), 4),
                    "second": item.get("sourceSecond"),
                }
            )

        prompt_data = {
            "persona": persona,
            "highlightSeconds": seconds[:10],
            "species": species_summary,
            "detections": detection_summary,
            "detectionCount": len(detections),
        }

        system_prompt = (
            f"{persona_hint} "
            "출력은 한국어 1~2문단, 최대 320자. 욕설/허위 정보 금지. "
            "마크다운 없이 평문만 출력한다."
        )
        user_prompt = (
            "다음 다이빙 분석 데이터를 기반으로 로그북 문장을 작성해줘.\n"
            f"{json.dumps(prompt_data, ensure_ascii=False)}"
        )

        headers = {
            "Authorization": f"Bearer {self.gms_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.logbook_llm_model,
            "messages": [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.logbook_llm_temperature,
            "max_tokens": self.logbook_llm_max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.logbook_llm_retries + 1):
            try:
                response = requests.post(
                    self.logbook_llm_url,
                    headers=headers,
                    json=body,
                    timeout=self.logbook_llm_timeout,
                )
                if response.status_code >= 400:
                    raise RuntimeError(f"llm request failed status={response.status_code} body={response.text[:400]}")

                payload = response.json()
                text = self._extract_chat_content(payload)
                normalized = re.sub(r"\s+", " ", text).strip()
                if not normalized:
                    raise RuntimeError("llm returned empty logbook")
                if len(normalized) > self.logbook_llm_max_chars:
                    normalized = normalized[: self.logbook_llm_max_chars].rstrip()
                return normalized
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.logbook_llm_retries:
                    time.sleep(attempt)

        raise RuntimeError(f"logbook llm failed: {last_error}")

    @staticmethod
    def _extract_chat_content(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        texts.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        texts.append(item["content"])
            return "\n".join(texts)
        return ""

    # =========================================================================
    # Vision LLM Pipeline - Analyze highlight images directly with Vision LLM
    # =========================================================================

    def _encode_image_to_base64(self, image_path: Path) -> str:
        """Encode image file to base64 string with resize for API efficiency."""
        try:
            with Image.open(str(image_path)) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.thumbnail(self.vision_image_max_size)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as exc:
            log.warning("image resize fallback path=%s error=%s", image_path, exc)
            with open(str(image_path), "rb") as fp:
                return base64.b64encode(fp.read()).decode("utf-8")

    def _analyze_frame_with_vision(self, image_path: Path, persona: str) -> str:
        """Analyze a single frame image using Vision LLM and return text description."""
        if not image_path.exists():
            return ""

        persona_prompts = {
            "기록형": (
                "당신은 전문 수중 객관적 기록관입니다. 사진 속의 해양 생물, 지형, 수중 상태(시야, 조류 등) 및 "
                "다이버의 스킬적 요소를 관찰 일지 형식으로 객관적이고 명확하게 묘사하세요. (반드시 200자 이내 핵심만 요약)"
            ),
            "예술가형": (
                "당신은 예술적인 수중 사진 작가입니다. 빛의 산란, 물결의 질감, 산호와 생물들의 아름다운 색감 조화를 "
                "감각적이고 화려한 문체로 시적인 느낌이 나게 묘사하세요. (반드시 200자 이내 핵심만 묘사)"
            ),
            "실속형": (
                "당신은 효율성을 중시하는 다이빙 가이드입니다. 사진에 등장하는 핵심 타겟 생물 이름, 주요 지형지물 1개, "
                "그리고 현재 다이버의 위치 정보 등 가장 중요한 팩트 3가지만을 불필요한 미사여구 없이 간결하게 나열하세요. "
                "(반드시 150자 이내)"
            ),
            "블로거형": (
                "당신은 트렌디한 스쿠버다이빙 인플루언서입니다. 사진 속 상황을 SNS에 올릴 귀엽고 재치 있는 말투"
                "(이모지 듬뿍)로 다이빙의 즐거운 순간을 생동감 있게 자랑하듯 묘사하세요. "
                "(반드시 200자 이내, 해시태그 포함)"
            ),
        }

        system_prompt = persona_prompts.get(persona, persona_prompts["기록형"])
        base64_image = self._encode_image_to_base64(image_path)
        model_name = self.vision_llm_model

        try:
            if "claude" in model_name.lower():
                return self._call_anthropic_vision(model_name, system_prompt, base64_image)
            else:
                return self._call_openai_vision(model_name, system_prompt, base64_image)
        except Exception as exc:
            log.warning("vision analysis failed image=%s model=%s error=%s", image_path, model_name, exc)
            return ""

    def _call_openai_vision(self, model_name: str, system_prompt: str, base64_image: str) -> str:
        """Call OpenAI Vision API via GMS proxy."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.gms_key}",
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "developer", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "제공된 이미지를 분석하세요."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                },
            ],
        }

        response = requests.post(self.logbook_llm_url, headers=headers, json=payload, timeout=self.vision_llm_timeout)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        raise ValueError(f"OpenAI response structure error: {result}")

    def _call_anthropic_vision(self, model_name: str, system_prompt: str, base64_image: str) -> str:
        """Call Anthropic Vision API via GMS proxy."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.gms_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": model_name,
            "max_tokens": 8192,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": "제공된 이미지를 분석하세요."},
                    ],
                }
            ],
        }

        response = requests.post(self.anthropic_url, headers=headers, json=payload, timeout=self.vision_llm_timeout)
        response.raise_for_status()
        result = response.json()

        if "content" in result and len(result["content"]) > 0:
            return result["content"][0]["text"]
        raise ValueError(f"Anthropic response structure error: {result}")

    def _synthesize_logbook_from_descriptions(self, descriptions: List[str], persona: str) -> str:
        """Synthesize frame descriptions into a cohesive dive logbook."""
        valid_texts = [txt for txt in descriptions if txt and txt.strip()]
        if not valid_texts:
            return ""

        combined_texts = "\n\n".join([f"[Frame {i+1}]\n{txt}" for i, txt in enumerate(valid_texts)])

        synthesis_prompt = (
            "5개의 시계열 텍스트 소스를 바탕으로 다이빙 로그를 작성하세요. "
            "각 단계는 시간 순서로 연결되어야 하며, 전체적인 다이빙의 흐름이 끊기지 않도록 연결하세요. "
            "선택된 페르소나의 문체를 유지하여 하나의 다이빙 로그를 완성하세요."
        )
        system_prompt = "You are an expert dive logbook author."
        user_content = f"{synthesis_prompt}\n\n[입력 묘사 소스]\n{combined_texts}"

        model_name = self.synthesis_llm_model
        headers = {
            "Authorization": f"Bearer {self.gms_key}",
            "Content-Type": "application/json",
        }

        try:
            if "claude" in model_name.lower():
                # Anthropic API
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.gms_key,
                    "anthropic-version": "2023-06-01",
                }
                payload = {
                    "model": model_name,
                    "max_tokens": 8192,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_content}],
                }
                response = requests.post(self.anthropic_url, headers=headers, json=payload, timeout=self.logbook_llm_timeout)
                response.raise_for_status()
                result = response.json()
                if "content" in result and len(result["content"]) > 0:
                    return result["content"][0]["text"]
            else:
                # OpenAI API
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "developer", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                }
                response = requests.post(self.logbook_llm_url, headers=headers, json=payload, timeout=self.logbook_llm_timeout)
                response.raise_for_status()
                return self._extract_chat_content(response.json())
        except Exception as exc:
            log.warning("synthesis llm failed model=%s error=%s", model_name, exc)
            return ""

        return ""

    def _build_logbook_with_vision_llm(
        self,
        persona: str,
        highlights: List[Dict[str, Any]],
        species: List[Dict[str, Any]],
    ) -> str:
        """Build logbook using Vision LLM pipeline - analyze images then synthesize."""
        if not self.vision_llm_enabled or not highlights:
            return ""

        log.info("vision llm pipeline started persona=%s highlights=%d", persona, len(highlights))

        # Step 1: Analyze each highlight frame with Vision LLM
        descriptions: List[str] = []
        for idx, highlight in enumerate(highlights):
            local_path = highlight.get("localPath")
            if not local_path:
                descriptions.append("")
                continue

            image_path = Path(local_path)
            log.info("analyzing frame %d/%d path=%s", idx + 1, len(highlights), image_path.name)
            description = self._analyze_frame_with_vision(image_path, persona)
            descriptions.append(description)

        # Step 2: Synthesize descriptions into final logbook
        log.info("synthesizing logbook from %d frame descriptions", len([d for d in descriptions if d]))
        logbook = self._synthesize_logbook_from_descriptions(descriptions, persona)

        # Add species info if logbook is too short
        if logbook and len(logbook) < 100 and species:
            species_names = [s.get("name") or s.get("koreanName") for s in species[:3] if s.get("name") or s.get("koreanName")]
            if species_names:
                logbook += f" (관찰된 주요 생물: {', '.join(species_names)})"

        return logbook

    def _upload_result_to_s3(self, task_id: str, result_payload: Dict[str, Any]) -> None:
        # 1. 하이라이트 이미지 업로드
        highlights = result_payload.get("highlights", [])
        for idx, highlight in enumerate(highlights):
            local_path = highlight.get("localPath")
            if local_path and Path(local_path).exists():
                suffix = Path(local_path).suffix.lower() or ".jpg"
                s3_key = f"analysis/highlights/{task_id}/highlight_{idx:02d}{suffix}"
                try:
                    self.s3.upload_file(local_path, self.bucket, s3_key, ExtraArgs={"ContentType": "image/jpeg"})
                    highlight["s3Url"] = f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{s3_key}"
                    highlight["s3Key"] = s3_key
                    log.info("uploaded highlight task_id=%s idx=%d key=%s", task_id, idx, s3_key)
                except Exception as exc:
                    log.warning("failed to upload highlight task_id=%s idx=%d error=%s", task_id, idx, exc)

        # 2. 물고기 크롭 이미지 업로드
        detections = result_payload.get("detections", [])
        crops_dir = None
        for detection in detections:
            image_name = detection.get("imageName")
            if not image_name:
                continue

            # crops_dir 탐색 (최초 1회)
            if crops_dir is None:
                for highlight in highlights:
                    hl_path = highlight.get("localPath")
                    if hl_path:
                        potential_crops = Path(hl_path).parent.parent / "crops"
                        if potential_crops.exists():
                            crops_dir = potential_crops
                            break
                        # run_dir 내 crops 폴더 탐색
                        run_dir = Path(hl_path).parent
                        if run_dir.exists():
                            crops_dir = run_dir
                            break
                if crops_dir is None:
                    crops_dir = self.cfd_dir / "results" / "best_shots"

            # 크롭 이미지 찾기 및 업로드
            crop_path = None
            if crops_dir and crops_dir.exists():
                for sub in crops_dir.iterdir():
                    if sub.is_dir():
                        candidate = sub / image_name
                        if candidate.exists():
                            crop_path = candidate
                            break
                    elif sub.name == image_name:
                        crop_path = sub
                        break

            if crop_path and crop_path.exists():
                suffix = crop_path.suffix.lower() or ".jpg"
                s3_key = f"analysis/fish_crops/{task_id}/{crop_path.name}"
                try:
                    self.s3.upload_file(str(crop_path), self.bucket, s3_key, ExtraArgs={"ContentType": "image/jpeg"})
                    detection["s3Url"] = f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{s3_key}"
                    detection["s3Key"] = s3_key
                    log.info("uploaded fish crop task_id=%s name=%s key=%s", task_id, image_name, s3_key)
                except Exception as exc:
                    log.warning("failed to upload fish crop task_id=%s name=%s error=%s", task_id, image_name, exc)

        # 3. 기존 JSON 업로드 (마지막에 - URL 포함)
        result_key = f"analysis/result/{task_id}.json"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=result_key,
            Body=json.dumps(result_payload, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        result_payload["resultS3Key"] = result_key

    def _callback_processing(self, task_id: str, callback: Dict[str, Any], upload_session_id: str | None = None) -> None:
        url = callback.get("processing_url") or self._default_callback("/api/internal/v1/analysis/processing")
        body = {
            "taskId": task_id,
            "workerId": self.worker_id,
            "startedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if upload_session_id:
            body["uploadSessionId"] = upload_session_id
        self._post(url, body)

    def _callback_complete(self, task_id: str, result_payload: Dict[str, Any], callback: Dict[str, Any]) -> None:
        url = callback.get("complete_url") or self._default_callback("/api/internal/v1/analysis/complete")
        body = {
            "taskId": task_id,
            "uploadSessionId": result_payload.get("uploadSessionId"),
            "persona": result_payload.get("persona"),
            "logbook": result_payload.get("logbook"),
            "highlights": result_payload.get("highlights", []),
            "detections": result_payload.get("detections", []),
            "species": result_payload.get("species", []),
            "completedAt": result_payload.get("completedAt"),
            "resultS3Key": result_payload.get("resultS3Key"),
        }
        self._post(url, body)

    def _callback_fail(
        self,
        task_id: str,
        error_code: str,
        error_message: str,
        callback: Dict[str, Any],
        upload_session_id: str | None = None,
    ) -> None:
        url = callback.get("fail_url") or self._default_callback("/api/internal/v1/analysis/fail")
        body = {
            "taskId": task_id,
            "errorCode": error_code,
            "errorMessage": error_message[:1000],
            "failedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if upload_session_id:
            body["uploadSessionId"] = upload_session_id
        self._post(url, body)

    def _default_callback(self, path: str) -> str:
        if not self.callback_base_url:
            raise RuntimeError("missing callback url and BACKEND_CALLBACK_BASE_URL is empty")
        return f"{self.callback_base_url}{path}"

    def _post(self, url: str, body: Dict[str, Any]) -> None:
        raw = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        headers = {"Content-Type": "application/json"}

        if self.callback_secret:
            ts = str(int(time.time()))
            sig = hmac.new(
                self.callback_secret.encode("utf-8"),
                f"{ts}.{raw}".encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Timestamp"] = ts
            headers["X-Internal-Signature"] = sig

        last_error: Exception | None = None
        for attempt in range(1, self.callback_retry_count + 1):
            try:
                response = requests.post(
                    url,
                    data=raw.encode("utf-8"),
                    headers=headers,
                    timeout=self.callback_timeout,
                )
                if response.status_code < 300:
                    return

                last_error = RuntimeError(
                    f"callback failed status={response.status_code} body={response.text[:400]}"
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc

            if attempt < self.callback_retry_count:
                backoff = self.callback_retry_delay_seconds * attempt
                log.warning(
                    "callback retry url=%s attempt=%s/%s backoff=%ss error=%s",
                    url,
                    attempt,
                    self.callback_retry_count,
                    backoff,
                    last_error,
                )
                time.sleep(backoff)

        raise RuntimeError(f"callback failed after retries: {last_error}")

    def _visibility_heartbeat_loop(self, receipt_handle: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(self.visibility_heartbeat_seconds):
            try:
                self.sqs.change_message_visibility(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=self.visibility_timeout,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("failed to extend message visibility: %s", exc)

    def _tail(self, text: str, limit: int) -> str:
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[-limit:]


def main() -> int:
    worker = Worker()
    worker.run()
    log.info("worker stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
