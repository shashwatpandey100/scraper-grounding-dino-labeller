"""Stage 4: Organize into YOLO directory structure + generate quality report."""

import json
import logging
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import PipelineConfig

log = logging.getLogger("scraper")


def organize_for_yolo(raw_dataset_dir: str, yolo_output_dir: str, config: PipelineConfig) -> dict:
    """
    Reorganize cleaned images into YOLO training structure.

    Input:  dataset_raw/{Category}/{SKU_Name}/[images]
    Output: yolo_dataset/
            ├── train/images/
            ├── train/labels/
            ├── val/images/
            ├── val/labels/
            ├── dataset.yaml
            └── class_mapping.json
    """
    # Build class mapping (alphabetically sorted)
    sku_entries = []
    for category_dir in sorted(os.listdir(raw_dataset_dir)):
        cat_path = os.path.join(raw_dataset_dir, category_dir)
        if not os.path.isdir(cat_path):
            continue
        for sku_dir in sorted(os.listdir(cat_path)):
            sku_path = os.path.join(cat_path, sku_dir)
            if os.path.isdir(sku_path) and sku_dir != "rejected":
                sku_entries.append({
                    "class_id": len(sku_entries),
                    "sku_name": sku_dir,
                    "category": category_dir,
                    "source_path": sku_path,
                })

    if not sku_entries:
        log.warning("No SKU folders found in raw dataset directory")
        return {}

    # Create output dirs
    for split in ("train", "val"):
        os.makedirs(os.path.join(yolo_output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(yolo_output_dir, split, "labels"), exist_ok=True)

    # Build all copy+label tasks, then execute in parallel
    class_mapping = {}
    copy_tasks = []

    for sku_info in sku_entries:
        class_id = sku_info["class_id"]
        sku_name = sku_info["sku_name"]
        source_path = sku_info["source_path"]
        class_mapping[class_id] = sku_name

        # Load Grounding DINO annotations if available
        annotations = {}
        annotations_path = os.path.join(source_path, "annotations.json")
        if os.path.exists(annotations_path):
            try:
                with open(annotations_path) as f:
                    data = json.load(f)
                annotations = data.get("annotations", {})
            except (json.JSONDecodeError, KeyError):
                pass

        # Get all valid images (skip rejected folder and json files)
        images = [
            f
            for f in os.listdir(source_path)
            if f.endswith((".jpg", ".jpeg", ".png", ".webp"))
            and os.path.isfile(os.path.join(source_path, f))
        ]

        if not images:
            log.warning(f"No images found for {sku_name}")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * config.train_split)
        if split_idx == len(images) and len(images) > 1:
            split_idx = len(images) - 1

        for i, img_name in enumerate(images):
            split = "train" if i < split_idx else "val"
            ext = os.path.splitext(img_name)[1]
            new_name = f"{sku_name}_{i:04d}{ext}"

            src = os.path.join(source_path, img_name)
            dst_img = os.path.join(yolo_output_dir, split, "images", new_name)
            label_name = f"{sku_name}_{i:04d}.txt"
            dst_label = os.path.join(yolo_output_dir, split, "labels", label_name)

            ann = annotations.get(img_name)
            if ann and "bbox_yolo" in ann:
                xc, yc, bw, bh = ann["bbox_yolo"]
                label_content = f"{class_id} {xc} {yc} {bw} {bh}\n"
                is_fallback = ann.get("fallback", False)
            else:
                label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"
                is_fallback = True

            copy_tasks.append({
                "src": src, "dst_img": dst_img, "dst_label": dst_label,
                "label_content": label_content, "split": split, "is_fallback": is_fallback,
            })

    # Execute all copy+label tasks in parallel
    def _copy_and_label(task):
        shutil.copy2(task["src"], task["dst_img"])
        with open(task["dst_label"], "w") as f:
            f.write(task["label_content"])

    with ThreadPoolExecutor(max_workers=config.stage4_copy_workers) as pool:
        list(pool.map(_copy_and_label, copy_tasks))

    total_train = sum(1 for t in copy_tasks if t["split"] == "train")
    total_val = sum(1 for t in copy_tasks if t["split"] == "val")
    total_annotated = sum(1 for t in copy_tasks if not t["is_fallback"])
    total_fallback = sum(1 for t in copy_tasks if t["is_fallback"])

    # Write dataset.yaml
    yaml_lines = [
        f"path: {os.path.abspath(yolo_output_dir)}",
        "train: train/images",
        "val: val/images",
        "",
        f"nc: {len(class_mapping)}",
        "",
        "names:",
    ]
    for class_id in sorted(class_mapping.keys()):
        yaml_lines.append(f"  {class_id}: {class_mapping[class_id]}")

    yaml_content = "\n".join(yaml_lines) + "\n"
    with open(os.path.join(yolo_output_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)

    # Write class mapping JSON
    with open(os.path.join(yolo_output_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=2)

    log.info(
        f"Stage 4 complete: {len(class_mapping)} classes, "
        f"{total_train} train / {total_val} val images, "
        f"{total_annotated} DINO-annotated / {total_fallback} fallback labels"
    )
    return class_mapping


# ---------------------------------------------------------------------------
# Quality report generation
# ---------------------------------------------------------------------------

def generate_report(all_sku_stats: dict, output_path: str) -> dict:
    """Generate a quality report as both JSON and human-readable text."""
    kept_values = [s.get("kept", 0) for s in all_sku_stats.values()]
    sim_values = [
        s.get("avg_visual_similarity", 0)
        for s in all_sku_stats.values()
        if s.get("avg_visual_similarity", 0) > 0
    ]

    report = {
        "summary": {
            "total_skus": len(all_sku_stats),
            "total_images_kept": sum(kept_values),
            "total_images_rejected": sum(
                s.get("rejected", 0) for s in all_sku_stats.values()
            ),
            "total_duplicates_removed": sum(
                s.get("duplicates_removed", 0) for s in all_sku_stats.values()
            ),
            "skus_below_minimum": sum(
                1 for s in all_sku_stats.values() if s.get("below_minimum", False)
            ),
            "avg_images_per_sku": float(np.mean(kept_values)) if kept_values else 0,
            "median_images_per_sku": float(np.median(kept_values)) if kept_values else 0,
            "min_images_per_sku": int(np.min(kept_values)) if kept_values else 0,
            "max_images_per_sku": int(np.max(kept_values)) if kept_values else 0,
            "avg_visual_similarity": float(np.mean(sim_values)) if sim_values else 0,
        },
        "skus_needing_attention": [
            sku
            for sku, stats in all_sku_stats.items()
            if stats.get("below_minimum", False)
        ],
        "skus_with_errors": [
            sku for sku, stats in all_sku_stats.items() if stats.get("error")
        ],
        "per_sku_stats": all_sku_stats,
    }

    # Save JSON report
    os.makedirs(output_path, exist_ok=True)
    report_path = os.path.join(output_path, "quality_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print human-readable summary
    s = report["summary"]
    print("\n" + "=" * 60)
    print("SCRAPING & QUALITY GATE REPORT")
    print("=" * 60)
    print(f"Total SKUs processed:       {s['total_skus']}")
    print(f"Total images kept:          {s['total_images_kept']}")
    print(f"Total images rejected:      {s['total_images_rejected']}")
    print(f"Total duplicates removed:   {s['total_duplicates_removed']}")
    print(f"Avg images per SKU:         {s['avg_images_per_sku']:.1f}")
    print(f"Median images per SKU:      {s['median_images_per_sku']:.1f}")
    print(f"Min / Max per SKU:          {s['min_images_per_sku']} / {s['max_images_per_sku']}")
    print(f"Avg visual similarity:      {s['avg_visual_similarity']:.3f}")
    print(f"SKUs below minimum (10):    {s['skus_below_minimum']}")

    if report["skus_needing_attention"]:
        print(f"\nSKUs needing manual image addition:")
        for sku in report["skus_needing_attention"]:
            kept = all_sku_stats[sku].get("kept", 0)
            print(f"   - {sku} ({kept} images)")

    if report["skus_with_errors"]:
        print(f"\nSKUs with errors:")
        for sku in report["skus_with_errors"]:
            err = all_sku_stats[sku].get("error", "unknown")
            print(f"   - {sku}: {err}")

    print("=" * 60)
    print(f"Report saved to: {report_path}")

    return report
