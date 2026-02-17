#!/usr/bin/env python3
"""
FMCG SKU Image Scraper with CLIP-Based Quality Gate + Grounding DINO Annotation
================================================================================

Master orchestrator that runs all 5 stages:
  1. Direct downloads from CSV (main_image_url + alt_image_urls)
  2. Web scraping from free sources (DuckDuckGo, BigBasket, OpenFoodFacts)
  3. CLIP quality gate + perceptual deduplication
  3b. Grounding DINO auto-annotation (bounding boxes)
  4. Organize into YOLO directory structure + quality report

Usage:
    python main.py                                  # Run full pipeline
    python main.py --csv my_skus.csv                # Custom CSV path
    python main.py --stage 3                        # Run from stage 3 onwards
    python main.py --stage 3b                       # Run from annotation onwards
    python main.py --stage 4                        # Only organize + report
    python main.py --no-resume                      # Force re-download everything
    python main.py --no-annotate                    # Skip Grounding DINO annotation
    python main.py --dino-confidence 0.4            # Tune DINO confidence threshold
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd

from src.config import PipelineConfig
from src.utils import setup_logging

log = logging.getLogger("scraper")

VALID_STAGES = ["1", "2", "3", "3b", "4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FMCG SKU Image Scraper with CLIP Quality Gate + DINO Annotation"
    )
    parser.add_argument(
        "--csv", default="selected_100_skus.csv", help="Path to input CSV file"
    )
    parser.add_argument(
        "--output-dir", default="dataset_raw", help="Raw dataset output directory"
    )
    parser.add_argument(
        "--yolo-dir", default="yolo_dataset_100", help="YOLO dataset output directory"
    )
    parser.add_argument(
        "--stage",
        default="1",
        choices=VALID_STAGES,
        help="Start from this stage (1, 2, 3, 3b, 4). Earlier stages are skipped.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume — redownload/reprocess everything",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Skip Grounding DINO auto-annotation (labels will be full-image fallback)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=25,
        help="Max images to keep per SKU (default: 25)",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=10,
        help="Min images per SKU before flagging (default: 10)",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model name (default: openai/clip-vit-base-patch32)",
    )
    parser.add_argument(
        "--dino-model",
        default="IDEA-Research/grounding-dino-tiny",
        help="Grounding DINO model name (default: IDEA-Research/grounding-dino-tiny)",
    )
    parser.add_argument(
        "--dino-confidence",
        type=float,
        default=0.3,
        help="Grounding DINO confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--visual-sim-high",
        type=float,
        default=0.85,
        help="Visual similarity threshold for 'keep_high' (default: 0.85)",
    )
    parser.add_argument(
        "--visual-sim-medium",
        type=float,
        default=0.70,
        help="Visual similarity threshold for 'keep_medium' (default: 0.70)",
    )
    parser.add_argument(
        "--visual-sim-low",
        type=float,
        default=0.60,
        help="Visual similarity threshold for 'keep_low' (default: 0.60)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        input_csv=args.csv,
        output_dir=args.output_dir,
        yolo_output_dir=args.yolo_dir,
        max_images_per_sku=args.max_images,
        min_images_per_sku=args.min_images,
        clip_model_name=args.clip_model,
        dino_model_name=args.dino_model,
        dino_confidence_threshold=args.dino_confidence,
        visual_similarity_high=args.visual_sim_high,
        visual_similarity_medium=args.visual_sim_medium,
        visual_similarity_low=args.visual_sim_low,
        train_split=args.train_split,
        resume=not args.no_resume,
    )


def stage_order(stage: str) -> int:
    """Convert stage string to a sortable number."""
    return {"1": 1, "2": 2, "3": 3, "3b": 4, "4": 5}[stage]


def validate_csv(df: pd.DataFrame) -> None:
    """Validate that the CSV has the required columns."""
    required = {"product_name", "brand", "main_image_url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    log.info(f"CSV loaded: {len(df)} SKUs, columns: {list(df.columns)}")


def main():
    args = parse_args()
    config = build_config(args)
    setup_logging(config.log_file)

    start_stage = stage_order(args.stage)

    log.info("=" * 60)
    log.info("FMCG SKU Image Scraper — Starting Pipeline")
    log.info("=" * 60)
    log.info(f"CSV:          {config.input_csv}")
    log.info(f"Output:       {config.output_dir}")
    log.info(f"YOLO dir:     {config.yolo_output_dir}")
    log.info(f"Resume:       {config.resume}")
    log.info(f"Start stage:  {args.stage}")
    log.info(f"Max images:   {config.max_images_per_sku}")
    log.info(f"CLIP model:   {config.clip_model_name}")
    log.info(f"DINO model:   {config.dino_model_name}")
    log.info(f"DINO conf:    {config.dino_confidence_threshold}")
    log.info(f"Annotate:     {not args.no_annotate}")

    # Load CSV
    if not os.path.exists(config.input_csv):
        log.error(f"CSV file not found: {config.input_csv}")
        sys.exit(1)

    df = pd.read_csv(config.input_csv)
    validate_csv(df)

    start_time = time.time()

    # ── Stage 1: Direct Downloads ──
    if start_stage <= 1:
        log.info("\n" + "─" * 40)
        log.info("STAGE 1: Direct Downloads from CSV")
        log.info("─" * 40)
        from src.stage1_download import run_stage1

        run_stage1(df, config)

    # ── Stage 2: Web Scraping ──
    if start_stage <= 2:
        log.info("\n" + "─" * 40)
        log.info("STAGE 2: Web Scraping (Free Sources)")
        log.info("─" * 40)
        from src.stage2_scrape import run_stage2

        run_stage2(df, config)

    # ── Stage 3: CLIP Quality Gate ──
    all_quality_stats = {}
    if start_stage <= 3:
        log.info("\n" + "─" * 40)
        log.info("STAGE 3: CLIP Quality Gate + Dedup")
        log.info("─" * 40)
        from src.stage3_quality import run_stage3

        all_quality_stats = run_stage3(df, config)

    # ── Stage 3b: Grounding DINO Auto-Annotation ──
    if start_stage <= 4 and not args.no_annotate:
        log.info("\n" + "─" * 40)
        log.info("STAGE 3b: Grounding DINO Auto-Annotation")
        log.info("─" * 40)
        from src.stage3b_annotate import run_stage3b

        run_stage3b(df, config)

    # ── Stage 4: YOLO Organization + Report ──
    log.info("\n" + "─" * 40)
    log.info("STAGE 4: YOLO Organization + Report")
    log.info("─" * 40)
    from src.stage4_organize import generate_report, organize_for_yolo

    organize_for_yolo(config.output_dir, config.yolo_output_dir, config)

    if all_quality_stats:
        generate_report(all_quality_stats, config.yolo_output_dir)

    elapsed = time.time() - start_time
    log.info(f"\nTotal pipeline time: {elapsed / 60:.1f} minutes")
    log.info("Done!")


if __name__ == "__main__":
    main()
