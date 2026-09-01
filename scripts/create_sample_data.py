"""
scripts/create_sample_data.py — Generate sample GeoTIFF tiles for testing and UI demonstration.

Creates sample 2-channel SAR GeoTIFFs (VV, VH) and matching 1-channel hand label GeoTIFFs
with valid WGS84 EPSG:4326 geospatial transforms in data/processed/ and data/samples/.
"""

import os
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import rasterio
from rasterio.transform import from_bounds
from ml.config import cfg

SAMPLES = [
    {
        "country": "Pakistan",
        "chip_id": "9928",
        "suffix": "S1Hand",
        "bounds": (68.0, 26.5, 68.25, 26.75),
        "center_lat": 26.625,
        "center_lon": 68.125,
        "location_name": "Indus River Basin, Sindh, Pakistan",
    },
    {
        "country": "Bolivia",
        "chip_id": "103757",
        "suffix": "S1Hand",
        "bounds": (-64.5, -16.5, -64.25, -16.25),
        "center_lat": -16.375,
        "center_lon": -64.375,
        "location_name": "Beni River Floodplain, Bolivia",
    },
    {
        "country": "USA",
        "chip_id": "10234",
        "suffix": "S1Hand",
        "bounds": (-90.5, 32.3, -90.25, 32.55),
        "center_lat": 32.425,
        "center_lon": -90.375,
        "location_name": "Mississippi River Inundation Zone, USA",
    },
    {
        "country": "Nigeria",
        "chip_id": "8812",
        "suffix": "S1Hand",
        "bounds": (6.5, 7.5, 6.75, 7.75),
        "center_lat": 7.625,
        "center_lon": 6.625,
        "location_name": "Benue River Confluence, Lokoja, Nigeria",
    },
]


def generate_synthetic_sar_and_label(height=256, width=256, seed=42):
    np.random.seed(seed)
    # Generate realistic SAR dB backscatter: VV ~ [-25, -5], VH ~ [-30, -10]
    # Create river / water channels with lower backscatter (darker)
    x = np.linspace(-2, 2, width)
    y = np.linspace(-2, 2, height)
    xx, yy = np.meshgrid(x, y)
    
    # River curve
    river_dist = np.abs(yy - 0.5 * np.sin(2 * xx) - 0.2 * xx)
    water_mask = (river_dist < 0.35).astype(np.float32)
    
    # Add random flood patches
    flood_patches = ((xx - 0.5)**2 + (yy - 0.5)**2 < 0.25).astype(np.float32)
    flood_mask = np.clip(water_mask + flood_patches, 0, 1)

    # SAR VV channel (water is dark, ~ -24 dB; land is bright, ~ -10 dB)
    vv = np.where(flood_mask > 0.5, -24.0 + np.random.normal(0, 1.5, (height, width)),
                                    -10.0 + np.random.normal(0, 3.0, (height, width)))
    # SAR VH channel
    vh = np.where(flood_mask > 0.5, -28.0 + np.random.normal(0, 1.5, (height, width)),
                                    -16.0 + np.random.normal(0, 3.0, (height, width)))
    
    sar = np.stack([vv, vh], axis=0).astype(np.float32)  # (2, 256, 256)
    
    # Label mask: 1 = flood, 0 = dry land
    label = flood_mask.astype(np.float32)  # (256, 256)
    
    # Generate SynEO RGB: 3 channels in [0, 1]
    # Water = dark blue/cyan, Land = green/brown
    eo_r = np.where(flood_mask > 0.5, 0.05, 0.25 + 0.1 * np.random.rand(height, width))
    eo_g = np.where(flood_mask > 0.5, 0.20, 0.45 + 0.1 * np.random.rand(height, width))
    eo_b = np.where(flood_mask > 0.5, 0.40, 0.20 + 0.1 * np.random.rand(height, width))
    syneo = np.stack([eo_r, eo_g, eo_b], axis=0).astype(np.float32)

    return sar, label, syneo


def main():
    print("=" * 60)
    print("GENERATING SAMPLE GEOTIFFS FOR FLOODSENSE")
    print("=" * 60)

    s1_dir = cfg.S1_DIR
    lbl_dir = cfg.LABEL_HAND
    syneo_dir = cfg.SYNEO_DIR
    sample_dir = cfg.DATA_DIR / "samples"

    for d in [s1_dir, lbl_dir, syneo_dir, sample_dir]:
        d.mkdir(parents=True, exist_ok=True)

    metadata_catalog = []

    for i, item in enumerate(SAMPLES):
        name = f"{item['country']}_{item['chip_id']}_{item['suffix']}"
        s1_path = s1_dir / f"{name}.tif"
        lbl_name = f"{item['country']}_{item['chip_id']}_LabelHand.tif"
        lbl_path = lbl_dir / lbl_name
        syneo_name = f"{item['country']}_{item['chip_id']}_SynEO.npy"
        syneo_path = syneo_dir / syneo_name
        
        # Also copy to data/samples for easy standalone demo
        sample_s1_path = sample_dir / f"{name}.tif"

        sar, label, syneo = generate_synthetic_sar_and_label(seed=42 + i)
        
        transform = from_bounds(*item['bounds'], 256, 256)
        crs = "EPSG:4326"

        # Write SAR GeoTIFF (2 channels)
        with rasterio.open(
            str(s1_path),
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=2,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(sar[0], 1)
            dst.set_band_description(1, "VV")
            dst.write(sar[1], 2)
            dst.set_band_description(2, "VH")

        with rasterio.open(str(sample_s1_path), "w", driver="GTiff", height=256, width=256, count=2, dtype="float32", crs=crs, transform=transform) as dst:
            dst.write(sar[0], 1)
            dst.write(sar[1], 2)

        # Write Label GeoTIFF (1 channel)
        with rasterio.open(
            str(lbl_path),
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(label[np.newaxis, ...])

        # Save SynEO .npy
        np.save(str(syneo_path), syneo)

        print(f"  Created sample {i+1}: {name}")
        print(f"    SAR:   {s1_path}")
        print(f"    Label: {lbl_path}")
        print(f"    SynEO: {syneo_path}")

        metadata_catalog.append({
            "sample_id": f"{item['country']}_{item['chip_id']}",
            "country": item["country"],
            "chip_id": item["chip_id"],
            "location_name": item["location_name"],
            "center_lat": item["center_lat"],
            "center_lon": item["center_lon"],
            "bounds": list(item["bounds"]),
            "crs": crs,
            "dimensions": "256x256",
            "channels": 2,
            "s1_path": str(s1_path),
            "label_path": str(lbl_path),
            "syneo_path": str(syneo_path),
            "has_ground_truth": True,
            "has_syneo": True,
        })

    # Save sample catalog JSON
    catalog_json = sample_dir / "samples_manifest.json"
    import json
    with open(catalog_json, "w") as fp:
        json.dump(metadata_catalog, fp, indent=2)

    print(f"\nSaved sample manifest to {catalog_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
