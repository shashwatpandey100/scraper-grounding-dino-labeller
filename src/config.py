"""Pipeline configuration and shared constants."""

import os
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    input_csv: str = "selected_100_skus.csv"
    output_dir: str = "dataset_raw"
    yolo_output_dir: str = "yolo_dataset_100"
    max_images_per_sku: int = 25
    min_images_per_sku: int = 10
    clip_model_name: str = "openai/clip-vit-base-patch32"
    visual_similarity_high: float = 0.85
    visual_similarity_medium: float = 0.70
    visual_similarity_low: float = 0.60
    text_similarity_medium: float = 0.22
    text_similarity_low: float = 0.25
    quality_threshold_low: float = 0.5
    dedup_hamming_threshold: int = 12
    train_split: float = 0.8
    max_ddg_results_per_query: int = 30
    download_timeout: int = 10
    download_retries: int = 3
    ddg_delay_min: float = 1.5
    ddg_delay_max: float = 3.0
    ecom_delay_min: float = 2.0
    ecom_delay_max: float = 5.0
    max_image_dimension: int = 1024
    dino_model_name: str = "IDEA-Research/grounding-dino-tiny"
    dino_confidence_threshold: float = 0.3
    log_file: str = "download_errors.log"
    resume: bool = True
    # Performance tuning
    stage1_download_workers: int = 16
    clip_batch_size: int = 16
    dino_batch_size: int = 8
    stage4_copy_workers: int = 8


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]


def get_random_headers() -> dict:
    """Return HTTP headers with a random User-Agent."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }


def get_amazon_headers() -> dict:
    """Return comprehensive browser-mimicking headers for Amazon India."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7,hi;q=0.6",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://www.amazon.in/",
    }


# Image content types we accept
VALID_IMAGE_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp",
}

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
