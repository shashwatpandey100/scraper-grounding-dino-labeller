"""Stage 2: Scrape additional product images from the open web (free methods only)."""

import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import PipelineConfig, get_amazon_headers, get_random_headers
from .utils import (
    download_image,
    extract_image_urls_from_json,
    get_sku_folder,
    is_valid_image_url,
    url_to_filename,
)

log = logging.getLogger("scraper")

# Track consecutive failures per domain to back off
_domain_failures: dict[str, int] = {}
# Track consecutive DDG timeouts across SKUs to detect sustained rate-limiting
_ddg_consecutive_timeouts: int = 0
_DDG_TIMEOUT_SKIP_THRESHOLD = 5  # Skip DDG entirely after this many consecutive timeouts


def detect_packaging_type(product_name: str, variant_text: str) -> str:
    """Infer packaging type from product name/variant."""
    text = f"{product_name} {variant_text}".lower()
    if any(w in text for w in ["tetra", "carton"]):
        return "tetra pack"
    if any(w in text for w in ["ml", "litre", "liter", "juice", "drink", "water", "shampoo", "oil"]):
        return "bottle"
    if any(w in text for w in ["gm", "gram", "chips", "biscuit", "cookie", "namkeen", "snack"]):
        return "packet"
    if any(w in text for w in ["can", "330ml"]):
        return "can"
    if any(w in text for w in ["soap", "bar"]):
        return "soap bar"
    if any(w in text for w in ["toothpaste", "cream", "gel", "face"]):
        return "tube"
    return "pack"


def generate_search_queries(brand: str, product_name: str, variant_text: str) -> list[str]:
    """Generate multiple search queries optimized for finding clean product images."""
    brand = brand.strip()
    product = product_name.strip()
    variant = variant_text.strip() if variant_text else ""
    packaging = detect_packaging_type(product, variant)

    # Avoid duplicating brand if product_name already starts with it
    if product.lower().startswith(brand.lower()):
        full_name = product
    else:
        full_name = f"{brand} {product}"

    queries = []

    # Query 1: Exact product for e-commerce results
    queries.append(f"{full_name} {variant} product image")

    # Query 2: With packaging type
    queries.append(f"{full_name} {variant} {packaging}")

    # Query 3: Target Indian e-commerce sites
    queries.append(
        f"{full_name} {variant} site:blinkit.com OR site:bigbasket.com OR site:amazon.in"
    )

    # Query 4: With India context for disambiguation
    if variant:
        queries.append(f"{full_name} {variant} india")

    # Query 5: Without variant for more general results
    queries.append(f"{full_name} {packaging} india")

    return queries[:5]


# ---------------------------------------------------------------------------
# DuckDuckGo Image Search
# ---------------------------------------------------------------------------

def search_ddg_images(
    brand: str,
    product_name: str,
    variant_text: str,
    config: PipelineConfig,
) -> list[dict]:
    """
    Search DuckDuckGo for product images using the duckduckgo-search library.
    Returns list of dicts with url, thumbnail, source, title, query keys.
    """
    global _ddg_consecutive_timeouts

    # If DDG has been timing out consistently, skip it entirely
    if _ddg_consecutive_timeouts >= _DDG_TIMEOUT_SKIP_THRESHOLD:
        log.info("DDG skipped (sustained timeouts — likely rate-limited)")
        return []

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        log.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return []

    queries = generate_search_queries(brand, product_name, variant_text)
    all_results = []
    seen_urls = set()
    timeouts_this_sku = 0

    with DDGS() as ddgs:
        for query in queries:
            try:
                results = ddgs.images(
                    keywords=query,
                    region="in-en",
                    safesearch="moderate",
                    size=None,
                    type_image="photo",
                    max_results=config.max_ddg_results_per_query,
                )
                for r in results:
                    img_url = r.get("image", "")
                    if img_url and img_url not in seen_urls:
                        seen_urls.add(img_url)
                        all_results.append({
                            "url": img_url,
                            "thumbnail": r.get("thumbnail", ""),
                            "source": r.get("url", ""),
                            "title": r.get("title", ""),
                            "query": query,
                        })
                # Success — reset the global timeout counter
                _ddg_consecutive_timeouts = 0
            except Exception as e:
                err_str = str(e).lower()
                log.warning(f"DDG search failed for '{query}': {e}")

                is_timeout = "timeout" in err_str
                is_ratelimit = "429" in err_str or "ratelimit" in err_str

                if is_timeout or is_ratelimit:
                    _ddg_consecutive_timeouts += 1
                    timeouts_this_sku += 1
                    # If every query for this SKU timed out, stop trying
                    if timeouts_this_sku >= 3:
                        log.warning(
                            f"DDG: 3 consecutive timeouts for this SKU, "
                            f"skipping remaining queries (global count: {_ddg_consecutive_timeouts})"
                        )
                        break
                    if is_ratelimit:
                        log.warning("DDG rate limit detected, backing off 60s")
                        time.sleep(60)
                continue

            delay = random.uniform(config.ddg_delay_min, config.ddg_delay_max)
            time.sleep(delay)

    return all_results


# ---------------------------------------------------------------------------
# BigBasket scraping
# ---------------------------------------------------------------------------

def scrape_bigbasket_images(
    brand: str, product_name: str, variant_text: str, config: PipelineConfig
) -> list[str]:
    """Scrape product images from BigBasket's public search page."""
    domain = "bigbasket.com"
    if _domain_failures.get(domain, 0) >= 3:
        return []

    urls = []
    try:
        query = f"{brand} {product_name} {variant_text}"
        search_url = f"https://www.bigbasket.com/ps/?q={requests.utils.quote(query)}"
        resp = requests.get(search_url, headers=get_random_headers(), timeout=config.download_timeout)
        if not resp.ok:
            _domain_failures[domain] = _domain_failures.get(domain, 0) + 1
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src", "") or img.get("data-src", "")
            if src and any(x in src for x in ["product", "pd-", "img-", "media"]):
                if src.startswith("//"):
                    src = "https:" + src
                if is_valid_image_url(src):
                    urls.append(src)
        _domain_failures[domain] = 0
    except Exception as e:
        log.warning(f"BigBasket scrape failed: {e}")
        _domain_failures[domain] = _domain_failures.get(domain, 0) + 1

    time.sleep(random.uniform(config.ecom_delay_min, config.ecom_delay_max))
    return urls


# ---------------------------------------------------------------------------
# Open Food Facts API (free, no key)
# ---------------------------------------------------------------------------

def search_openfoodfacts(brand: str, product_name: str) -> list[str]:
    """Search Open Food Facts API for product images."""
    urls = []
    try:
        search_url = (
            f"https://world.openfoodfacts.org/cgi/search.pl?"
            f"search_terms={requests.utils.quote(f'{brand} {product_name}')}"
            f"&search_simple=1&json=1&page_size=5"
        )
        resp = requests.get(search_url, timeout=10)
        if resp.ok:
            data = resp.json()
            for product in data.get("products", []):
                for key in ("image_url", "image_front_url", "image_front_small_url"):
                    val = product.get(key)
                    if val and isinstance(val, str) and val.startswith("http"):
                        urls.append(val)
                # Also grab other image URLs dynamically
                for key, val in product.items():
                    if "image" in key and isinstance(val, str) and val.startswith("http"):
                        if val not in urls:
                            urls.append(val)
    except Exception as e:
        log.warning(f"OpenFoodFacts search failed: {e}")

    return urls


# ---------------------------------------------------------------------------
# Flipkart scraping (SSR HTML)
# ---------------------------------------------------------------------------

def scrape_flipkart_images(
    brand: str, product_name: str, variant_text: str, config: PipelineConfig
) -> list[str]:
    """Scrape product images from Flipkart search results."""
    domain = "flipkart.com"
    if _domain_failures.get(domain, 0) >= 3:
        return []

    urls = []
    try:
        query = f"{brand} {product_name} {variant_text}".strip()
        search_url = f"https://www.flipkart.com/search?q={requests.utils.quote(query)}&marketplace=FLIPKART"
        resp = requests.get(
            search_url, headers=get_random_headers(), timeout=config.download_timeout
        )
        if not resp.ok:
            _domain_failures[domain] = _domain_failures.get(domain, 0) + 1
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src", "") or img.get("data-src", "")
            if src and "rukminim" in src and "flixcart.com" in src:
                # Upgrade to higher quality image
                src = re.sub(r"/\d+/\d+/", "/416/416/", src)
                src = re.sub(r"\?q=\d+", "?q=90", src)
                if src.startswith("//"):
                    src = "https:" + src
                if src not in urls:
                    urls.append(src)

        _domain_failures[domain] = 0
    except Exception as e:
        log.warning(f"Flipkart scrape failed: {e}")
        _domain_failures[domain] = _domain_failures.get(domain, 0) + 1

    time.sleep(random.uniform(config.ecom_delay_min, config.ecom_delay_max))
    return urls


# ---------------------------------------------------------------------------
# Blinkit scraping (internal API)
# ---------------------------------------------------------------------------

def scrape_blinkit_images(
    brand: str, product_name: str, variant_text: str, config: PipelineConfig
) -> list[str]:
    """Scrape product images from Blinkit via internal search API."""
    domain = "blinkit.com"
    if _domain_failures.get(domain, 0) >= 3:
        return []

    urls = []
    query = f"{brand} {product_name} {variant_text}".strip()

    # Try Blinkit's internal search API
    api_url = "https://blinkit.com/v6/search/products"
    headers = get_random_headers()
    headers.update({
        "Accept": "application/json",
        "Referer": "https://blinkit.com/",
    })
    params = {
        "q": query,
        "lat": "28.6139",
        "lon": "77.2090",
    }

    try:
        resp = requests.get(
            api_url, headers=headers, params=params, timeout=config.download_timeout
        )
        if resp.ok:
            try:
                data = resp.json()
                found = extract_image_urls_from_json(
                    data, ["cdn.grofers.com", "blinkit"]
                )
                urls.extend(found)
                _domain_failures[domain] = 0
            except ValueError:
                _domain_failures[domain] = _domain_failures.get(domain, 0) + 1
        else:
            _domain_failures[domain] = _domain_failures.get(domain, 0) + 1
    except Exception as e:
        log.warning(f"Blinkit scrape failed: {e}")
        _domain_failures[domain] = _domain_failures.get(domain, 0) + 1

    time.sleep(random.uniform(config.ddg_delay_min, config.ddg_delay_max))
    # Deduplicate
    return list(dict.fromkeys(urls))


# ---------------------------------------------------------------------------
# Swiggy Instamart scraping (internal API)
# ---------------------------------------------------------------------------

def scrape_swiggy_instamart_images(
    brand: str, product_name: str, variant_text: str, config: PipelineConfig
) -> list[str]:
    """Scrape product images from Swiggy Instamart via internal API."""
    domain = "swiggy.com"
    if _domain_failures.get(domain, 0) >= 3:
        return []

    urls = []
    query = f"{brand} {product_name} {variant_text}".strip()

    headers = get_random_headers()
    headers.update({
        "Accept": "application/json",
        "Referer": "https://www.swiggy.com/instamart",
    })

    # Try Swiggy Instamart search API
    api_endpoints = [
        f"https://www.swiggy.com/api/instamart/search?q={requests.utils.quote(query)}&lat=28.6139&lng=77.2090",
        f"https://www.swiggy.com/dapi/instamart/search?q={requests.utils.quote(query)}&lat=28.6139&lng=77.2090",
    ]

    for api_url in api_endpoints:
        try:
            resp = requests.get(
                api_url, headers=headers, timeout=config.download_timeout
            )
            if resp.ok:
                try:
                    data = resp.json()
                    found = extract_image_urls_from_json(
                        data,
                        ["instamart-media-assets.swiggy.com", "media-assets.swiggy.com", "res.cloudinary.com/swiggy"],
                    )
                    urls.extend(found)
                    _domain_failures[domain] = 0
                    break  # Success on this endpoint, stop trying others
                except ValueError:
                    continue
            # Non-OK but not a hard failure — try next endpoint
        except Exception as e:
            log.warning(f"Swiggy Instamart scrape failed ({api_url}): {e}")
            continue

    if not urls:
        _domain_failures[domain] = _domain_failures.get(domain, 0) + 1

    time.sleep(random.uniform(config.ddg_delay_min, config.ddg_delay_max))
    return list(dict.fromkeys(urls))


# ---------------------------------------------------------------------------
# Amazon India scraping (SSR HTML, aggressive anti-bot)
# ---------------------------------------------------------------------------

def scrape_amazon_in_images(
    brand: str, product_name: str, variant_text: str, config: PipelineConfig
) -> list[str]:
    """Scrape product images from Amazon India search results."""
    domain = "amazon.in"
    if _domain_failures.get(domain, 0) >= 3:
        return []

    urls = []
    try:
        query = f"{brand} {product_name} {variant_text}".strip()
        search_url = f"https://www.amazon.in/s?k={requests.utils.quote(query)}"
        resp = requests.get(
            search_url, headers=get_amazon_headers(), timeout=15
        )

        if not resp.ok:
            _domain_failures[domain] = _domain_failures.get(domain, 0) + 1
            return []

        text_lower = resp.text.lower()
        if "captcha" in text_lower or "automated access" in text_lower:
            log.warning("Amazon CAPTCHA detected, backing off")
            _domain_failures[domain] = _domain_failures.get(domain, 0) + 1
            return []

        soup = BeautifulSoup(resp.text, "html.parser")

        # Primary: product thumbnails with class "s-image"
        for img in soup.find_all("img", class_="s-image"):
            src = img.get("src", "")
            if src and "m.media-amazon.com" in src:
                # Upscale: replace size suffix for better quality
                src = re.sub(r"\._[A-Z][A-Z0-9_]+_\.", "._AC_SL500_.", src)
                if src not in urls:
                    urls.append(src)

        # Fallback: any img from Amazon CDN
        if not urls:
            for img in soup.find_all("img"):
                src = img.get("src", "")
                if src and "m.media-amazon.com" in src and "/images/I/" in src:
                    src = re.sub(r"\._[A-Z][A-Z0-9_]+_\.", "._AC_SL500_.", src)
                    if src not in urls:
                        urls.append(src)

        _domain_failures[domain] = 0
    except Exception as e:
        log.warning(f"Amazon India scrape failed: {e}")
        _domain_failures[domain] = _domain_failures.get(domain, 0) + 1

    # Longer delay for Amazon
    time.sleep(random.uniform(3.0, 8.0))
    return urls


# ---------------------------------------------------------------------------
# Stage 2 orchestration per SKU
# ---------------------------------------------------------------------------

def _scrape_source(func, args, prefix):
    """Wrapper to call a scraper function and tag results with a prefix."""
    try:
        result = func(*args)
        # DDG returns list of dicts, others return list of str
        if result and isinstance(result[0], dict):
            return [(r["url"], prefix) for r in result]
        return [(url, prefix) for url in result]
    except Exception as e:
        log.warning(f"Source {prefix} failed: {e}")
        return []


def scrape_additional_images(row: pd.Series, config: PipelineConfig) -> dict:
    """
    Scrape additional images for one SKU from multiple free sources.
    Sources are scraped CONCURRENTLY, then images are downloaded in parallel.
    Returns stats dict.
    """
    sku_folder = get_sku_folder(row, config.output_dir)
    os.makedirs(sku_folder, exist_ok=True)

    brand = str(row.get("brand", "")).strip()
    product_name = str(row.get("product_name", "")).strip()
    variant_text = str(row.get("variant_text", "") or "").strip()

    # Check how many images we already have (resume support)
    existing = [
        f for f in os.listdir(sku_folder)
        if f.endswith((".jpg", ".jpeg", ".png", ".webp"))
        and os.path.isfile(os.path.join(sku_folder, f))
        and f != "reference.jpg"
    ]
    if config.resume and len(existing) >= config.max_images_per_sku:
        log.info(f"  {product_name}: already has {len(existing)} images, skipping scrape")
        return {
            "ddg": 0, "bigbasket": 0, "openfoodfacts": 0,
            "flipkart": 0, "blinkit": 0, "swiggy": 0, "amazon": 0,
            "skipped": True,
        }

    stats = {
        "ddg": 0, "bigbasket": 0, "openfoodfacts": 0,
        "flipkart": 0, "blinkit": 0, "swiggy": 0, "amazon": 0,
        "skipped": False,
    }

    # ── Scrape all 7 sources CONCURRENTLY ──
    # Each source has its own internal delays so they don't interfere.
    # Running in parallel: total time ≈ max(source times) instead of sum.
    sources = [
        (search_ddg_images, (brand, product_name, variant_text, config), "ddg"),
        (scrape_bigbasket_images, (brand, product_name, variant_text, config), "bb"),
        (search_openfoodfacts, (brand, product_name), "off"),
        (scrape_flipkart_images, (brand, product_name, variant_text, config), "fk"),
        (scrape_blinkit_images, (brand, product_name, variant_text, config), "blinkit"),
        (scrape_swiggy_instamart_images, (brand, product_name, variant_text, config), "swiggy"),
        (scrape_amazon_in_images, (brand, product_name, variant_text, config), "amzn"),
    ]

    all_urls: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=7) as pool:
        futures = {
            pool.submit(_scrape_source, func, args, prefix): prefix
            for func, args, prefix in sources
        }
        for future in as_completed(futures):
            all_urls.extend(future.result())

    # Deduplicate by URL
    seen = set()
    unique_urls = []
    for url, prefix in all_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append((url, prefix))

    # ── Download images in PARALLEL (I/O-bound) ──
    prefix_to_key = {
        "ddg": "ddg", "bb": "bigbasket", "off": "openfoodfacts",
        "fk": "flipkart", "blinkit": "blinkit", "swiggy": "swiggy", "amzn": "amazon",
    }

    def _download_one(item):
        i, url, prefix = item
        filename = url_to_filename(url, prefix=prefix, index=i)
        save_path = os.path.join(sku_folder, filename)
        success = download_image(url, save_path, config)
        return (prefix, success)

    download_items = [(i, url, prefix) for i, (url, prefix) in enumerate(unique_urls)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for prefix, success in pool.map(_download_one, download_items):
            if success:
                source_key = prefix_to_key.get(prefix, prefix)
                stats[source_key] = stats.get(source_key, 0) + 1

    return stats


def run_stage2(df: pd.DataFrame, config: PipelineConfig) -> dict:
    """
    Run Stage 2 for all SKUs in the dataframe.
    Returns dict mapping SKU names to scraping stats.
    """
    from tqdm import tqdm

    all_stats = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stage 2: Web Scraping"):
        sku_name = str(row.get("product_name", "unknown"))
        stats = scrape_additional_images(row, config)
        all_stats[sku_name] = stats
        if not stats.get("skipped"):
            log.info(
                f"  {sku_name}: DDG={stats['ddg']}, BB={stats['bigbasket']}, "
                f"OFF={stats['openfoodfacts']}, FK={stats['flipkart']}, "
                f"BL={stats['blinkit']}, SW={stats['swiggy']}, AMZ={stats['amazon']}"
            )

    source_keys = ("ddg", "bigbasket", "openfoodfacts", "flipkart", "blinkit", "swiggy", "amazon")
    total_scraped = sum(
        sum(s.get(k, 0) for k in source_keys)
        for s in all_stats.values()
    )
    log.info(f"Stage 2 complete: {total_scraped} additional images scraped")
    return all_stats
