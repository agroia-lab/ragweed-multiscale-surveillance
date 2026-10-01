#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Instituto de Investigaciones Agropecuarias (INIA), Chile
"""
AMBEL Geo Export for Lentejas/Ragweed Pipeline
===============================================

Exports GPS coordinates + AMBEL detection counts to GeoPackage for QGIS visualization.
Adapted from an earlier crop-detection geo-export script for the AMBEL (ragweed) class.

Usage:
    # Ground photos
    python scripts/03_inference/export_ambel_geo.py \
        --images /path/to/ground/photos \
        --summary /path/to/summary.json \
        --output-name lencu_ground_ambel_20241220 \
        --output-dir /path/to/outputs/geo_exports \
        --device-type ground

    # Drone photos
    python scripts/03_inference/export_ambel_geo.py \
        --images /path/to/drone/images \
        --summary /path/to/summary.json \
        --output-name lencu_drone_ambel_20241224 \
        --output-dir /path/to/outputs/geo_exports \
        --device-type drone

    # Merge existing GeoPackages
    python scripts/03_inference/export_ambel_geo.py \
        --merge /path/to/geo_exports/lencu_ground_ambel_20241220.gpkg \
                /path/to/geo_exports/lencu_drone_ambel_20241224.gpkg \
        --output-name lencu_combined_ambel \
        --output-dir /path/to/outputs/geo_exports
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path using portable resolution
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root))

from scripts.utils.paths import get_project_root
from scripts.utils.geo_data_utils import extract_gps_coordinates

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
except ImportError:
    print("ERROR: geopandas required. Install: pip install geopandas")
    sys.exit(1)


def get_image_files(directory: Path) -> list:
    """Get all image files from directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.JPG', '.JPEG', '.HEIC'}
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
    return sorted(files)


def extract_gps_batch(image_dir: Path, verbose: bool = True) -> dict:
    """
    Extract GPS coordinates from all images in directory.

    Returns:
        Dict mapping image stem to {latitude, longitude, altitude}
    """
    gps_data = {}
    image_files = get_image_files(image_dir)

    if verbose:
        print(f"  Extracting GPS from {len(image_files)} images...")

    no_gps_count = 0
    for i, img_path in enumerate(image_files):
        if verbose and (i + 1) % 50 == 0:
            print(f"    Progress: {i + 1}/{len(image_files)}")

        lat, lon, alt, lat_ref, lon_ref = extract_gps_coordinates(str(img_path))

        if lat is not None and lon is not None:
            gps_data[img_path.stem] = {
                'latitude': lat,
                'longitude': lon,
                'altitude': alt if alt is not None else 0,
            }
        else:
            no_gps_count += 1

    if verbose and no_gps_count > 0:
        print(f"  WARNING: {no_gps_count} images had no GPS data")

    return gps_data


def load_ambel_summary(summary_path: Path) -> dict:
    """
    Load detection summary and return AMBEL counts per image.

    Returns:
        Dict mapping image stem to AMBEL count (int).
    """
    with open(summary_path) as f:
        summary = json.load(f)

    results = {}

    if 'per_image_results' in summary:
        for item in summary['per_image_results']:
            img_name = item.get('image', '')
            stem = Path(img_name).stem
            counts = item.get('sahi_by_class', {})
            results[stem] = counts.get('AMBEL', 0)

    elif 'per_image' in summary:
        for img_name, counts in summary['per_image'].items():
            stem = Path(img_name).stem
            results[stem] = counts.get('AMBEL', 0)

    else:
        print(f"WARNING: Unknown summary format in {summary_path}")

    return results


def export_geopackage(
    gps_data: dict,
    ambel_counts: dict,
    output_name: str,
    output_dir: Path,
    device_type: str = "ground",
    dataset_label: str = None,
) -> Path:
    """
    Build GeoDataFrame from GPS + AMBEL counts and export to GeoPackage.

    Returns:
        Path to created .gpkg file, or None on failure.
    """
    records = []

    for stem, gps in gps_data.items():
        lat = gps.get('latitude')
        lon = gps.get('longitude')
        alt = gps.get('altitude', 0)

        if lat is None or lon is None:
            continue

        nr_ambel = ambel_counts.get(stem, 0)

        record = {
            'image': stem,
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'nr_ambel': nr_ambel,
            'device': device_type,
            'geometry': Point(lon, lat),
        }

        if dataset_label:
            record['dataset'] = dataset_label

        records.append(record)

    if not records:
        print("ERROR: No valid GPS records to export")
        return None

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = output_dir / f"{output_name}.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")

    return gpkg_path


def merge_geopackages(gpkg_paths: list, output_name: str, output_dir: Path) -> Path:
    """
    Merge multiple GeoPackages into a single combined file.
    Infers a 'dataset' field from each filename if not present.

    Returns:
        Path to merged .gpkg file.
    """
    frames = []
    for p in gpkg_paths:
        p = Path(p)
        if not p.exists():
            print(f"WARNING: Skipping missing file: {p}")
            continue
        gdf = gpd.read_file(p)
        if 'dataset' not in gdf.columns:
            # Infer dataset label from filename
            name = p.stem
            if 'ground' in name.lower():
                gdf['dataset'] = 'ground'
            elif 'drone' in name.lower():
                gdf['dataset'] = 'drone'
            else:
                gdf['dataset'] = name
        frames.append(gdf)

    if not frames:
        print("ERROR: No valid GeoPackages to merge")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs=frames[0].crs)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_name}.gpkg"
    combined.to_file(out_path, driver="GPKG")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Export AMBEL detection results to GeoPackage with GPS coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Export mode
    parser.add_argument('--images', type=str,
                        help='Directory with images (for GPS extraction)')
    parser.add_argument('--summary', type=str,
                        help='Path to summary.json from SAHI inference')
    parser.add_argument('--output-name', type=str, required=True,
                        help='Output base name (without extension)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for .gpkg files')
    parser.add_argument('--device-type', type=str, default='ground',
                        choices=['ground', 'drone'],
                        help='Device type (ground or drone)')

    # Merge mode
    parser.add_argument('--merge', nargs='+', type=str,
                        help='Merge multiple .gpkg files into one combined file')

    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()
    verbose = not args.quiet

    # ---- MERGE MODE ----
    if args.merge:
        if verbose:
            print("=" * 60)
            print("  AMBEL GEO MERGE")
            print("=" * 60)
            for p in args.merge:
                print(f"  Input: {p}")
            print()

        out_path = merge_geopackages(
            gpkg_paths=args.merge,
            output_name=args.output_name,
            output_dir=Path(args.output_dir),
        )

        if out_path:
            gdf = gpd.read_file(out_path)
            if verbose:
                print(f"\n  Merged GeoPackage: {out_path}")
                print(f"  Total features: {len(gdf)}")
                print(f"  Total AMBEL: {gdf['nr_ambel'].sum():,}")
                if 'dataset' in gdf.columns:
                    for ds, grp in gdf.groupby('dataset'):
                        print(f"    {ds}: {len(grp)} features, {grp['nr_ambel'].sum():,} AMBEL")
                print("=" * 60)
        return

    # ---- EXPORT MODE ----
    if not args.images or not args.summary:
        parser.error("--images and --summary are required for export mode (or use --merge)")

    images_dir = Path(args.images)
    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)

    if not summary_path.exists():
        print(f"ERROR: Summary file not found: {summary_path}")
        sys.exit(1)

    if verbose:
        print("=" * 60)
        print("  AMBEL GEO EXPORT")
        print("=" * 60)
        print(f"  Images: {images_dir}")
        print(f"  Summary: {summary_path}")
        print(f"  Output: {args.output_name}.gpkg")
        print(f"  Device: {args.device_type}")
        print()

    # Step 1: Extract GPS
    if verbose:
        print("[1/3] EXTRACTING GPS DATA")

    gps_data = extract_gps_batch(images_dir, verbose=verbose)
    total_images = len(get_image_files(images_dir))

    coverage = len(gps_data) / total_images if total_images > 0 else 0
    if verbose:
        print(f"  GPS coverage: {len(gps_data)}/{total_images} ({coverage:.1%})")

    if len(gps_data) == 0:
        print("\nERROR: No GPS data found. Cannot create GeoPackage.")
        sys.exit(1)

    # Step 2: Load AMBEL counts
    if verbose:
        print("\n[2/3] LOADING AMBEL DETECTION COUNTS")

    ambel_counts = load_ambel_summary(summary_path)

    if verbose:
        print(f"  Loaded counts for {len(ambel_counts)} images")
        total_ambel = sum(ambel_counts.values())
        print(f"  Total AMBEL detections: {total_ambel:,}")

    # Step 3: Export GeoPackage
    if verbose:
        print("\n[3/3] EXPORTING GEOPACKAGE")

    gpkg_path = export_geopackage(
        gps_data=gps_data,
        ambel_counts=ambel_counts,
        output_name=args.output_name,
        output_dir=output_dir,
        device_type=args.device_type,
        dataset_label=args.device_type,
    )

    if gpkg_path:
        gdf = gpd.read_file(gpkg_path)

        if verbose:
            print("\n" + "=" * 60)
            print("  EXPORT COMPLETE")
            print("=" * 60)
            print(f"  GeoPackage: {gpkg_path}")
            print(f"  Features:   {len(gdf)}")
            print(f"  AMBEL total: {gdf['nr_ambel'].sum():,}")
            print(f"  AMBEL mean:  {gdf['nr_ambel'].mean():.1f}")
            print(f"  AMBEL max:   {gdf['nr_ambel'].max()}")
            print(f"  Images with detections: {(gdf['nr_ambel'] > 0).sum()}/{len(gdf)}")
            print("=" * 60)
    else:
        print("\nERROR: Failed to create GeoPackage")
        sys.exit(1)


if __name__ == '__main__':
    main()
