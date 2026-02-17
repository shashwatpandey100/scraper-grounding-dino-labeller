"""Stage 3: CLIP-based quality gate + perceptual deduplication."""

import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import PipelineConfig

log = logging.getLogger("scraper")

# Module-level model state (loaded once, reused)
_clip_model = None
_clip_processor = None
_device = None


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32") -> None:
    """Load the CLIP model and processor once into module globals."""
    global _clip_model, _clip_processor, _device

    if _clip_model is not None:
        return  # Already loaded

    from transformers import CLIPModel, CLIPProcessor

    _device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    log.info(f"Loading CLIP model '{model_name}' on device '{_device}'...")
    _clip_model = CLIPModel.from_pretrained(model_name).to(_device)
    _clip_processor = CLIPProcessor.from_pretrained(model_name)
    _clip_model.eval()
    log.info("CLIP model loaded.")


def _to_tensor(output) -> torch.Tensor:
    """Extract a plain tensor from a model output (handles transformers 5.x)."""
    if isinstance(output, torch.Tensor):
        return output
    # BaseModelOutputWithPooling or similar — grab the pooler_output or first attr
    if hasattr(output, "image_embeds"):
        return output.image_embeds
    if hasattr(output, "text_embeds"):
        return output.text_embeds
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    # Last resort: index into it
    return output[0]


def get_image_embedding(image: Image.Image) -> torch.Tensor:
    """Get L2-normalized CLIP embedding for a single image."""
    inputs = _clip_processor(images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        raw = _clip_model.get_image_features(**inputs)
    embedding = _to_tensor(raw)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu()


def get_image_embeddings_batch(
    images: list[Image.Image], batch_size: int = 16
) -> torch.Tensor:
    """
    Get L2-normalized CLIP embeddings for a list of images in batches.
    Returns tensor of shape (N, embed_dim).
    """
    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = _clip_processor(images=batch, return_tensors="pt").to(_device)
        with torch.no_grad():
            raw = _clip_model.get_image_features(**inputs)
        embeddings = _to_tensor(raw)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def get_text_embedding(text: str) -> torch.Tensor:
    """Get L2-normalized CLIP embedding for a text description."""
    inputs = _clip_processor(text=[text], return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        raw = _clip_model.get_text_features(**inputs)
    embedding = _to_tensor(raw)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu()


# ---------------------------------------------------------------------------
# Image quality heuristics
# ---------------------------------------------------------------------------

def check_image_quality(img: Image.Image) -> float:
    """Basic image quality heuristics. Returns 0.0–1.0."""
    score = 1.0
    w, h = img.size

    # Too small (likely a thumbnail or icon)
    if w < 150 or h < 150:
        score -= 0.5

    # Too large aspect ratio (likely a banner or collage)
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > 3.0:
        score -= 0.4

    # Check if image is mostly one color (blank/placeholder)
    img_array = np.array(img.convert("RGB"))
    std_dev = np.std(img_array)
    if std_dev < 20:
        score -= 0.5

    # Check if too dark or too bright on average
    mean_brightness = np.mean(img_array)
    if mean_brightness < 30 or mean_brightness > 245:
        score -= 0.3

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Multi-signal scoring
# ---------------------------------------------------------------------------

def _score_with_embedding(
    candidate_img: Image.Image,
    candidate_emb: torch.Tensor,
    reference_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    config: PipelineConfig,
) -> dict:
    """Score a candidate using a pre-computed CLIP embedding."""
    if candidate_emb.dim() == 1:
        candidate_emb = candidate_emb.unsqueeze(0)

    visual_similarity = torch.cosine_similarity(reference_embedding, candidate_emb).item()
    text_similarity = torch.cosine_similarity(text_embedding, candidate_emb).item()
    quality_score = check_image_quality(candidate_img)

    combined_score = (0.55 * visual_similarity) + (0.30 * text_similarity) + (0.15 * quality_score)

    decision = "reject"
    confidence = "low"

    if visual_similarity >= config.visual_similarity_high:
        decision = "keep_high"
        confidence = "high"
    elif visual_similarity >= config.visual_similarity_medium and text_similarity >= config.text_similarity_medium:
        decision = "keep_medium"
        confidence = "medium"
    elif (
        visual_similarity >= config.visual_similarity_low
        and text_similarity >= config.text_similarity_low
        and quality_score >= config.quality_threshold_low
    ):
        decision = "keep_low"
        confidence = "low"
    else:
        decision = "reject"
        confidence = "high"

    return {
        "visual_similarity": round(visual_similarity, 4),
        "text_similarity": round(text_similarity, 4),
        "quality_score": round(quality_score, 4),
        "combined_score": round(combined_score, 4),
        "decision": decision,
        "confidence": confidence,
    }


def score_candidate_image(
    candidate_img: Image.Image,
    reference_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    config: PipelineConfig,
) -> dict:
    """Score a candidate image (computes embedding on the fly). Kept for compatibility."""
    candidate_emb = get_image_embedding(candidate_img)
    return _score_with_embedding(candidate_img, candidate_emb, reference_embedding, text_embedding, config)


# ---------------------------------------------------------------------------
# Perceptual deduplication
# ---------------------------------------------------------------------------

def perceptual_hash(img: Image.Image, hash_size: int = 16) -> str:
    """Compute a perceptual hash (average hash) for deduplication."""
    img_small = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.array(img_small)
    avg = pixels.mean()
    hash_bits = (pixels > avg).flatten()
    hash_int = 0
    for bit in hash_bits:
        hash_int = (hash_int << 1) | int(bit)
    return hex(hash_int)


def hamming_distance(hash1: str, hash2: str) -> int:
    """Count differing bits between two perceptual hashes."""
    h1 = int(hash1, 16)
    h2 = int(hash2, 16)
    return bin(h1 ^ h2).count("1")


def deduplicate_images(image_paths: list[str], threshold: int = 12) -> list[str]:
    """
    Remove near-duplicate images. Keep the highest resolution version.
    threshold: max hamming distance to consider as duplicate (lower = stricter).
    """
    hashes: dict[str, str] = {}  # hash -> path
    keep: list[str] = []

    for path in image_paths:
        try:
            img = Image.open(path)
            h = perceptual_hash(img)

            is_dup = False
            for existing_hash, existing_path in list(hashes.items()):
                if hamming_distance(h, existing_hash) < threshold:
                    # Keep the higher resolution one
                    existing_img = Image.open(existing_path)
                    if (img.size[0] * img.size[1]) > (
                        existing_img.size[0] * existing_img.size[1]
                    ):
                        hashes.pop(existing_hash)
                        hashes[h] = path
                        keep = [p for p in keep if p != existing_path]
                        keep.append(path)
                    is_dup = True
                    break

            if not is_dup:
                hashes[h] = path
                keep.append(path)
        except Exception:
            continue

    return keep


# ---------------------------------------------------------------------------
# Full quality gate pipeline per SKU
# ---------------------------------------------------------------------------

def run_quality_gate_for_sku(
    sku_folder: str,
    brand: str,
    product_name: str,
    variant_text: str,
    config: PipelineConfig,
) -> dict:
    """
    Run the full quality gate pipeline for one SKU.

    1. Load reference image and compute CLIP embedding
    2. Score every candidate image
    3. Deduplicate
    4. Select top images up to max_images
    5. Move rejects to a 'rejected' subfolder

    Returns stats dict.
    """
    scores_path = os.path.join(sku_folder, "quality_scores.json")

    # Resume: if quality scores already exist, load and return
    if config.resume and os.path.exists(scores_path):
        try:
            with open(scores_path) as f:
                cached = json.load(f)
            log.info(f"  Using cached quality scores for {os.path.basename(sku_folder)}")
            return cached.get("stats", {})
        except (json.JSONDecodeError, KeyError):
            pass

    reference_path = os.path.join(sku_folder, "reference.jpg")
    if not os.path.exists(reference_path):
        log.warning(f"No reference image in {sku_folder}")
        return {"error": "No reference image found", "kept": 0}

    # Load reference embedding
    ref_img = Image.open(reference_path).convert("RGB")
    ref_embedding = get_image_embedding(ref_img)

    # Create text embedding for this product
    text_desc = f"a photo of {brand} {product_name} {variant_text} product packaging"
    text_emb = get_text_embedding(text_desc)

    # Score all candidate images (everything except reference.jpg)
    all_files = [
        f
        for f in os.listdir(sku_folder)
        if f.endswith((".jpg", ".jpeg", ".png", ".webp"))
        and os.path.isfile(os.path.join(sku_folder, f))
        and f != "reference.jpg"
    ]

    # Phase 1: Load all candidate images
    loaded_images = []
    loaded_fnames = []
    loaded_fpaths = []
    for fname in all_files:
        fpath = os.path.join(sku_folder, fname)
        try:
            img = Image.open(fpath).convert("RGB")
            loaded_images.append(img)
            loaded_fnames.append(fname)
            loaded_fpaths.append(fpath)
        except Exception as e:
            log.warning(f"Failed to open {fpath}: {e}")

    # Phase 2: Batch-compute ALL candidate embeddings at once
    candidates = []
    if loaded_images:
        all_embeddings = get_image_embeddings_batch(loaded_images, batch_size=config.clip_batch_size)

        # Phase 3: Score each using pre-computed embeddings
        for i, (img, fname, fpath) in enumerate(zip(loaded_images, loaded_fnames, loaded_fpaths)):
            try:
                scores = _score_with_embedding(img, all_embeddings[i], ref_embedding, text_emb, config)
                scores["path"] = fpath
                scores["filename"] = fname
                candidates.append(scores)
            except Exception as e:
                log.warning(f"Failed to score {fpath}: {e}")

    # Separate keeps and rejects
    keeps = [c for c in candidates if c["decision"].startswith("keep")]
    rejects = [c for c in candidates if c["decision"] == "reject"]

    # Sort keeps by combined score (best first)
    keeps.sort(key=lambda x: x["combined_score"], reverse=True)

    # Deduplicate among keeps
    keep_paths = [c["path"] for c in keeps]
    deduped_paths = deduplicate_images(keep_paths, threshold=config.dedup_hamming_threshold)

    # Trim to max_images (always include reference)
    final_paths = deduped_paths[: config.max_images_per_sku]

    # Move rejects to rejected subfolder
    rejected_dir = os.path.join(sku_folder, "rejected")
    os.makedirs(rejected_dir, exist_ok=True)

    final_paths_set = set(final_paths)
    for c in candidates:
        if c["path"] not in final_paths_set:
            dest = os.path.join(rejected_dir, c["filename"])
            if os.path.exists(c["path"]):
                shutil.move(c["path"], dest)

    kept_count = len(final_paths) + 1  # +1 for reference
    stats = {
        "total_candidates": len(candidates),
        "kept": kept_count,
        "rejected": len(candidates) - len(final_paths),
        "duplicates_removed": len(keep_paths) - len(deduped_paths),
        "avg_visual_similarity": float(
            np.mean([c["visual_similarity"] for c in keeps]) if keeps else 0
        ),
        "below_minimum": kept_count < config.min_images_per_sku,
    }

    # Cache quality scores for resume
    cache_data = {
        "stats": stats,
        "kept_files": [os.path.basename(p) for p in final_paths],
        "scores": [
            {k: v for k, v in c.items() if k != "path"} for c in candidates
        ],
    }
    with open(scores_path, "w") as f:
        json.dump(cache_data, f, indent=2)

    return stats


def run_stage3(df: pd.DataFrame, config: PipelineConfig) -> dict:
    """
    Run Stage 3 (CLIP quality gate) for all SKUs.
    Returns dict mapping SKU names to their quality stats.
    """
    from tqdm import tqdm
    from .utils import get_sku_folder

    load_clip_model(config.clip_model_name)

    all_stats = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stage 3: Quality Gate"):
        sku_folder = get_sku_folder(row, config.output_dir)
        sku_name = str(row.get("product_name", "unknown"))
        brand = str(row.get("brand", "")).strip()
        product_name = str(row.get("product_name", "")).strip()
        variant_text = str(row.get("variant_text", "") or "").strip()

        if not os.path.isdir(sku_folder):
            log.warning(f"SKU folder not found: {sku_folder}")
            all_stats[sku_name] = {"error": "folder not found", "kept": 0}
            continue

        stats = run_quality_gate_for_sku(
            sku_folder, brand, product_name, variant_text, config
        )
        all_stats[sku_name] = stats
        log.info(
            f"  {sku_name}: kept={stats.get('kept', 0)}, "
            f"rejected={stats.get('rejected', 0)}, "
            f"dedup={stats.get('duplicates_removed', 0)}, "
            f"avg_sim={stats.get('avg_visual_similarity', 0):.3f}"
        )

    total_kept = sum(s.get("kept", 0) for s in all_stats.values())
    total_rejected = sum(s.get("rejected", 0) for s in all_stats.values())
    log.info(f"Stage 3 complete: {total_kept} images kept, {total_rejected} rejected")
    return all_stats
