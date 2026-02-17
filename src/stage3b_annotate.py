"""Stage 3b: Auto-annotate images with Grounding DINO bounding boxes."""

import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import PipelineConfig

log = logging.getLogger("scraper")

# Module-level model state (loaded once)
_dino_model = None
_dino_processor = None
_device = None

# Text prompt for detecting product packaging in images
DETECTION_PROMPT = "product packaging."


def load_grounding_dino(
    model_name: str = "IDEA-Research/grounding-dino-tiny",
) -> None:
    """Load Grounding DINO model and processor once into module globals."""
    global _dino_model, _dino_processor, _device

    if _dino_model is not None:
        return

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    _device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    log.info(f"Loading Grounding DINO '{model_name}' on device '{_device}'...")
    _dino_processor = AutoProcessor.from_pretrained(model_name)
    _dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(
        _device
    )
    _dino_model.eval()
    log.info("Grounding DINO loaded.")


def detect_product_bbox(
    image: Image.Image,
    confidence_threshold: float = 0.3,
) -> list[dict]:
    """
    Run Grounding DINO on a single image to detect product bounding boxes.

    Returns list of dicts with keys: bbox_xyxy, bbox_yolo, confidence, label.
    bbox_yolo is normalized (x_center, y_center, width, height) ready for YOLO format.
    """
    w, h = image.size

    inputs = _dino_processor(
        images=image, text=DETECTION_PROMPT, return_tensors="pt"
    ).to(_device)

    with torch.no_grad():
        outputs = _dino_model(**inputs)

    results = _dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=confidence_threshold,
        target_sizes=[(h, w)],
    )[0]

    detections = []
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box

        # Clamp to image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        # Convert to YOLO normalized format: x_center, y_center, width, height
        bw = x2 - x1
        bh = y2 - y1
        if bw < 1 or bh < 1:
            continue

        x_center = (x1 + x2) / 2.0 / w
        y_center = (y1 + y2) / 2.0 / h
        norm_w = bw / w
        norm_h = bh / h

        detections.append(
            {
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_yolo": [
                    round(float(x_center), 6),
                    round(float(y_center), 6),
                    round(float(norm_w), 6),
                    round(float(norm_h), 6),
                ],
                "confidence": round(float(score), 4),
            }
        )

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def detect_product_bboxes_batch(
    images: list[Image.Image],
    confidence_threshold: float = 0.3,
    batch_size: int = 8,
) -> list[list[dict]]:
    """
    Run Grounding DINO on a batch of images. All use the same text prompt.
    Returns a list (one per image) of lists of detection dicts.
    """
    all_results = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        batch_sizes = [(img.size[1], img.size[0]) for img in batch_images]  # (h, w)
        text_prompts = [DETECTION_PROMPT] * len(batch_images)

        inputs = _dino_processor(
            images=batch_images,
            text=text_prompts,
            return_tensors="pt",
            padding=True,
        ).to(_device)

        with torch.no_grad():
            outputs = _dino_model(**inputs)

        batch_results = _dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=confidence_threshold,
            target_sizes=batch_sizes,
        )

        for results, image in zip(batch_results, batch_images):
            w, h = image.size
            detections = []
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                bw = x2 - x1
                bh = y2 - y1
                if bw < 1 or bh < 1:
                    continue
                x_center = (x1 + x2) / 2.0 / w
                y_center = (y1 + y2) / 2.0 / h
                norm_w = bw / w
                norm_h = bh / h
                detections.append({
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "bbox_yolo": [
                        round(float(x_center), 6),
                        round(float(y_center), 6),
                        round(float(norm_w), 6),
                        round(float(norm_h), 6),
                    ],
                    "confidence": round(float(score), 4),
                })
            detections.sort(key=lambda d: d["confidence"], reverse=True)
            all_results.append(detections)

    return all_results


def annotate_sku_folder(
    sku_folder: str,
    confidence_threshold: float = 0.3,
    resume: bool = True,
    dino_batch_size: int = 8,
) -> dict:
    """
    Run Grounding DINO on all kept images in one SKU folder.
    Uses batched inference for speed, with sequential fallback.

    Saves annotations to annotations.json in the SKU folder.
    Returns stats dict.
    """
    annotations_path = os.path.join(sku_folder, "annotations.json")

    # Resume: skip if already annotated
    if resume and os.path.exists(annotations_path):
        try:
            with open(annotations_path) as f:
                cached = json.load(f)
            log.info(f"  Using cached annotations for {os.path.basename(sku_folder)}")
            return cached.get("stats", {})
        except (json.JSONDecodeError, KeyError):
            pass

    image_files = [
        f
        for f in os.listdir(sku_folder)
        if f.endswith((".jpg", ".jpeg", ".png", ".webp"))
        and os.path.isfile(os.path.join(sku_folder, f))
    ]

    if not image_files:
        return {"annotated": 0, "no_detection": 0, "error": 0}

    all_annotations = {}
    stats = {"annotated": 0, "no_detection": 0, "error": 0}

    # Phase 1: Load all images
    loaded_images = []
    loaded_fnames = []
    for fname in image_files:
        fpath = os.path.join(sku_folder, fname)
        try:
            img = Image.open(fpath).convert("RGB")
            loaded_images.append(img)
            loaded_fnames.append(fname)
        except Exception as e:
            log.warning(f"Failed to open {fpath}: {e}")
            all_annotations[fname] = {
                "bbox_yolo": [0.5, 0.5, 1.0, 1.0],
                "confidence": 0.0, "num_detections": 0, "fallback": True,
            }
            stats["error"] += 1

    # Phase 2: Batch inference (with sequential fallback)
    if loaded_images:
        try:
            batch_detections = detect_product_bboxes_batch(
                loaded_images, confidence_threshold, batch_size=dino_batch_size
            )
        except Exception as e:
            log.warning(f"Batch DINO failed, falling back to sequential: {e}")
            batch_detections = []
            for img in loaded_images:
                try:
                    batch_detections.append(detect_product_bbox(img, confidence_threshold))
                except Exception:
                    batch_detections.append([])

        # Phase 3: Map detections to filenames
        for fname, detections in zip(loaded_fnames, batch_detections):
            if detections:
                best = detections[0]
                all_annotations[fname] = {
                    "bbox_yolo": best["bbox_yolo"],
                    "confidence": best["confidence"],
                    "num_detections": len(detections),
                }
                stats["annotated"] += 1
            else:
                all_annotations[fname] = {
                    "bbox_yolo": [0.5, 0.5, 1.0, 1.0],
                    "confidence": 0.0, "num_detections": 0, "fallback": True,
                }
                stats["no_detection"] += 1

    # Save to disk for resume and Stage 4
    cache = {"stats": stats, "annotations": all_annotations}
    with open(annotations_path, "w") as f:
        json.dump(cache, f, indent=2)

    return stats


def run_stage3b(df: pd.DataFrame, config: PipelineConfig) -> dict:
    """
    Run auto-annotation for all SKUs.
    Returns dict mapping SKU names to annotation stats.
    """
    from tqdm import tqdm

    from .utils import get_sku_folder

    load_grounding_dino()

    all_stats = {}
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Stage 3b: Auto-Annotation"
    ):
        sku_name = str(row.get("product_name", "unknown"))
        sku_folder = get_sku_folder(row, config.output_dir)

        if not os.path.isdir(sku_folder):
            log.warning(f"SKU folder not found: {sku_folder}")
            all_stats[sku_name] = {"error": "folder not found"}
            continue

        stats = annotate_sku_folder(
            sku_folder,
            confidence_threshold=config.dino_confidence_threshold,
            resume=config.resume,
            dino_batch_size=config.dino_batch_size,
        )
        all_stats[sku_name] = stats
        log.info(
            f"  {sku_name}: annotated={stats.get('annotated', 0)}, "
            f"no_det={stats.get('no_detection', 0)}, err={stats.get('error', 0)}"
        )

    total = sum(s.get("annotated", 0) for s in all_stats.values())
    fallbacks = sum(
        s.get("no_detection", 0) + s.get("error", 0) for s in all_stats.values()
    )
    log.info(
        f"Stage 3b complete: {total} images annotated, {fallbacks} used fallback bbox"
    )
    return all_stats
