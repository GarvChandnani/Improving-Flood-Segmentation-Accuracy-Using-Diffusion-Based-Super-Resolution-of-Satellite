"""
ml/data_loader.py — Dataset catalog discovery and asset download utilities.

Source: Notebook Cells 6 (download_asset) and 8 (collect_all_assets).

These functions work with the Sen1Floods11 STAC catalog JSON files
to build lists of SAR/label/S2 asset URLs, and download them resumably.
"""

import os
import json
import glob
import requests
from pathlib import Path
from typing import List, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Catalog discovery (Cell 8)
# ---------------------------------------------------------------------------

def collect_all_assets(pattern: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Collect all chip assets matching a Sen1Floods11 catalog glob pattern.

    Exact replica of notebook Cell 8:

        def collect_all_assets(pattern):
            results = []
            for path in glob.glob(pattern):
                d = json.load(open(path))
                results.append((d['properties']['country'],
                                 d['properties']['chip_id'],
                                 d['assets']))
            return results

    Args:
        pattern: Glob pattern pointing to STAC JSON files, e.g.
                 "/content/catalog_extracted/catalog/sen1floods11_hand_labeled_source/*/*.json"

    Returns:
        List of (country, chip_id, assets_dict) tuples.
        `assets_dict` maps asset names (e.g. "S1Hand") to dicts with an "href" key.

    Example usage:
        hand_source = collect_all_assets(f"{catalog_base}/sen1floods11_hand_labeled_source/*/*.json")
        hand_label  = collect_all_assets(f"{catalog_base}/sen1floods11_hand_labeled_label/*/*.json")
        weak_source = collect_all_assets(f"{catalog_base}/sen1floods11_weak_labeled_source/*/*.json")
        weak_label  = collect_all_assets(f"{catalog_base}/sen1floods11_weak_labeled_label/*/*.json")
    """
    results = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r") as f:
            d = json.load(f)
        results.append((
            d["properties"]["country"],
            d["properties"]["chip_id"],
            d["assets"],
        ))
    return results


# ---------------------------------------------------------------------------
# Asset download (Cell 6)
# ---------------------------------------------------------------------------

def download_asset(url: str, out_path: str, timeout: int = 60) -> None:
    """
    Download a single asset URL to a local file. Skips if already downloaded.

    Exact replica of notebook Cell 6:

        def download_asset(url, out_path):
            if os.path.exists(out_path):
                return
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(out_path, 'wb') as f:
                f.write(r.content)

    Args:
        url:      HTTP URL to download.
        out_path: Local file path to save to.
        timeout:  Request timeout in seconds.

    Raises:
        requests.exceptions.HTTPError on non-200 responses.
    """
    if os.path.exists(out_path):
        return  # Already downloaded — idempotent
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)


# ---------------------------------------------------------------------------
# Batch download helpers (Cell 12)
# ---------------------------------------------------------------------------

def download_all_s1_hand(hand_source: list, s1_dir: str) -> None:
    """Download all S1Hand SAR tiles from hand-labeled source catalog."""
    from tqdm import tqdm
    for country, chip_id, assets in tqdm(hand_source, desc="S1Hand"):
        out = os.path.join(s1_dir, f"{country}_{chip_id}_S1Hand.tif")
        download_asset(assets["S1Hand"]["href"], out)


def download_all_labels_hand(hand_label: list, label_dir: str) -> None:
    """Download all LabelHand flood masks."""
    from tqdm import tqdm
    for country, chip_id, assets in tqdm(hand_label, desc="LabelHand"):
        out = os.path.join(label_dir, f"{country}_{chip_id}_LabelHand.tif")
        download_asset(assets["LabelHand"]["href"], out)


def download_all_s2_hand(hand_source: list, s2_dir: str) -> List[Tuple[str, str]]:
    """
    Download S2Hand optical tiles. Returns list of (country, chip_id) that failed.

    From notebook Cell 12:
        for country, chip_id, assets in tqdm(hand_source, desc="S2Hand"):
            url = assets['S1Hand']['href'].replace('S1Hand', 'S2Hand')
            try: download_asset(url, ...)
            except requests.exceptions.HTTPError: failed_s2.append(...)
    """
    from tqdm import tqdm
    failed = []
    for country, chip_id, assets in tqdm(hand_source, desc="S2Hand"):
        url = assets["S1Hand"]["href"].replace("S1Hand", "S2Hand")
        out = os.path.join(s2_dir, f"{country}_{chip_id}_S2Hand.tif")
        try:
            download_asset(url, out)
        except requests.exceptions.HTTPError:
            failed.append((country, chip_id))
    return failed


def download_all_s1_weak(weak_source: list, s1_dir: str) -> None:
    """Download all S1Weak SAR tiles from weak-labeled source catalog."""
    from tqdm import tqdm
    for country, chip_id, assets in tqdm(weak_source, desc="S1Weak"):
        out = os.path.join(s1_dir, f"{country}_{chip_id}_S1Weak.tif")
        download_asset(assets["S1Weak"]["href"], out)


def download_all_weak_labels(weak_label: list, label_dir: str) -> None:
    """Download all Otsu weak flood labels."""
    from tqdm import tqdm
    for country, chip_id, assets in tqdm(weak_label, desc="WeakLabel"):
        out = os.path.join(label_dir, f"{country}_{chip_id}_S1OtsuLabelWeak.tif")
        download_asset(assets["S1OtsuLabelWeak"]["href"], out)
