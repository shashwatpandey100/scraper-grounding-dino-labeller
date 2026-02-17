"""Shared utility functions for the scraper pipeline."""

import hashlib
import logging
import os
import re
import time
from urllib.parse import urlparse

import requests
from PIL import Image

from .config import (
    VALID_IMAGE_CONTENT_TYPES,
    VALID_IMAGE_EXTENSIONS,
    PipelineConfig,
    get_random_headers,
)

log = logging.getLogger("scraper")


def setup_logging(log_file: str = "download_errors.log") -> None:
    """Configure logging to both console and file."""
    root = logging.getLogger("scraper")
    root.setLevel(logging.INFO)

    if root.handlers:
        return

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)


def sanitize_name(name: str) -> str:
    """Convert a product/category name into a safe directory name."""
    name = name.strip()
    name = re.sub(r"[^\w\s\-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def get_sku_folder(row, output_dir: str) -> str:
    """Build the SKU folder path from a CSV row."""
    category = sanitize_name(str(row.get("category", "Uncategorized")))
    product = sanitize_name(str(row["product_name"]))
    variant = sanitize_name(str(row.get("variant_text", "") or ""))
    sku_name = f"{product}_{variant}" if variant else product
    return os.path.join(output_dir, category, sku_name)


def get_sku_name(row) -> str:
    """Build a human-readable SKU name from a CSV row."""
    product = sanitize_name(str(row["product_name"]))
    variant = sanitize_name(str(row.get("variant_text", "") or ""))
    return f"{product}_{variant}" if variant else product


def is_valid_image_url(url: str) -> bool:
    """Check if a URL looks like it points to an image."""
    if not url or not url.startswith(("http://", "https://", "//")):
        return False
    parsed = urlparse(url)
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in VALID_IMAGE_EXTENSIONS):
        return True
    # URLs without file extensions may still be images (e.g., CDN URLs)
    if any(kw in url.lower() for kw in ["image", "img", "photo", "media", "cdn", "product"]):
        return True
    return True  # Be permissive; we validate content-type on download


def download_image(
    url: str,
    save_path: str,
    config: PipelineConfig,
    max_dim: int = None,
) -> bool:
    """
    Download an image from a URL and save it locally.

    - Retries with exponential backoff on failure.
    - Validates content type.
    - Resizes to max_dim on longest side.
    - Returns True on success, False on failure.
    """
    if max_dim is None:
        max_dim = config.max_image_dimension

    # Resume: skip if already downloaded
    if config.resume and os.path.exists(save_path):
        return True

    if url.startswith("//"):
        url = "https:" + url

    for attempt in range(config.download_retries):
        try:
            resp = requests.get(
                url,
                headers=get_random_headers(),
                timeout=config.download_timeout,
                stream=True,
            )

            if resp.status_code in (404, 403, 410):
                log.warning(f"HTTP {resp.status_code} for {url}")
                return False

            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if content_type and content_type not in VALID_IMAGE_CONTENT_TYPES:
                log.warning(f"Non-image content-type '{content_type}' for {url}")
                return False

            # Read image data
            img_data = resp.content
            if len(img_data) < 500:
                log.warning(f"Tiny response ({len(img_data)} bytes) for {url}")
                return False

            # Validate it's a real image and resize
            from io import BytesIO

            img = Image.open(BytesIO(img_data))
            img = img.convert("RGB")

            # Resize if needed
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path, "JPEG", quality=90)
            return True

        except (requests.RequestException, OSError, Image.UnidentifiedImageError) as e:
            wait = 2 ** attempt
            log.warning(
                f"Download attempt {attempt + 1}/{config.download_retries} failed for {url}: {e}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    log.error(f"All {config.download_retries} download attempts failed for {url}")
    return False


def url_to_filename(url: str, prefix: str = "img", index: int = 0) -> str:
    """Generate a deterministic filename from a URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{prefix}_{index:03d}_{url_hash}.jpg"


def extract_image_urls_from_json(data, cdn_patterns: list[str]) -> list[str]:
    """Recursively walk a JSON structure and extract image URLs matching CDN patterns."""
    urls: list[str] = []
    if isinstance(data, dict):
        for value in data.values():
            urls.extend(extract_image_urls_from_json(value, cdn_patterns))
    elif isinstance(data, list):
        for item in data:
            urls.extend(extract_image_urls_from_json(item, cdn_patterns))
    elif isinstance(data, str):
        if data.startswith("http") and any(p in data for p in cdn_patterns):
            urls.append(data)
    return urls
