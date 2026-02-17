"""Stage 1: Download images directly from CSV URLs (main_image_url + alt_image_urls)."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .config import PipelineConfig
from .utils import download_image, get_sku_folder

log = logging.getLogger("scraper")


def _download_task(task: dict, config: PipelineConfig) -> dict:
    """Execute a single download task. Returns the task dict with 'success' added."""
    task["success"] = download_image(task["url"], task["save_path"], config)
    return task


def run_stage1(df: pd.DataFrame, config: PipelineConfig) -> dict:
    """
    Run Stage 1 for all SKUs — downloads reference + alt images in parallel.

    Flattens all download tasks across all SKUs, then executes them
    concurrently with ThreadPoolExecutor. CDN downloads have no rate limits.
    """
    from tqdm import tqdm

    # Phase 1: Build all download tasks
    all_tasks = []
    for _, row in df.iterrows():
        sku_name = str(row.get("product_name", "unknown"))
        sku_folder = get_sku_folder(row, config.output_dir)
        os.makedirs(sku_folder, exist_ok=True)

        # Reference image
        main_url = str(row.get("main_image_url", "")).strip()
        if main_url and main_url.lower() != "nan":
            all_tasks.append({
                "sku_name": sku_name,
                "url": main_url,
                "save_path": os.path.join(sku_folder, "reference.jpg"),
                "type": "reference",
            })

        # Alt images
        alt_urls_raw = str(row.get("alt_image_urls", "")).strip()
        if alt_urls_raw and alt_urls_raw.lower() != "nan":
            alt_urls = [u.strip() for u in alt_urls_raw.split("|") if u.strip()]
            for i, url in enumerate(alt_urls):
                all_tasks.append({
                    "sku_name": sku_name,
                    "url": url,
                    "save_path": os.path.join(sku_folder, f"alt_{i + 1:03d}.jpg"),
                    "type": "alt",
                })

    log.info(f"Stage 1: {len(all_tasks)} total images to download across {len(df)} SKUs")

    # Phase 2: Execute all downloads in parallel
    all_stats = {}
    workers = config.stage1_download_workers

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_task, task, config): task for task in all_tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Stage 1: Direct Downloads"):
            task = future.result()
            sku = task["sku_name"]
            if sku not in all_stats:
                all_stats[sku] = {"reference": False, "alt_downloaded": 0, "alt_failed": 0}
            if task["type"] == "reference":
                all_stats[sku]["reference"] = task["success"]
                if not task["success"]:
                    log.error(f"CRITICAL: Failed to download reference image for {sku}: {task['url']}")
            else:
                if task["success"]:
                    all_stats[sku]["alt_downloaded"] += 1
                else:
                    all_stats[sku]["alt_failed"] += 1

    # Log summary
    total_ref = sum(1 for s in all_stats.values() if s["reference"])
    total_alt = sum(s["alt_downloaded"] for s in all_stats.values())
    log.info(
        f"Stage 1 complete: {total_ref}/{len(all_stats)} references, "
        f"{total_alt} alt images downloaded (workers={workers})"
    )
    return all_stats
