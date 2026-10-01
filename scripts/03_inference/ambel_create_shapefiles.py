# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Instituto de Investigaciones Agropecuarias (INIA), Chile
"""
AMBEL Adult — Create Geo-referenced Shapefiles
================================================
Reads the 3 inference CSVs and creates ESRI shapefiles:
  - ambel_sta_rosa_ground.shp
  - ambel_cato_maiz_ground.shp
  - ambel_trigo_corregidas_ground.shp
  - ambel_all_ground.shp (merged, with 'dataset' column)

Input:  CSVs written by ambel_sahi_inference_full.py (--csv-root)
Output: --output-dir (default: outputs/geo_exports/)
CRS: EPSG:4326

Usage:
    python scripts/03_inference/ambel_create_shapefiles.py \
        --csv-root outputs/ambel_adult_full --output-dir outputs/geo_exports
"""

import os
import argparse
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

_ap = argparse.ArgumentParser(description="Create geo-referenced shapefiles from AMBEL inference CSVs")
_ap.add_argument("--csv-root", default="outputs/ambel_adult_full",
                 help="Folder with the per-dataset CSVs from ambel_sahi_inference_full.py")
_ap.add_argument("--output-dir", default="outputs/geo_exports", help="Output folder for shapefiles")
_args = _ap.parse_args()

CSV_ROOT = _args.csv_root
GEO_EXPORTS = _args.output_dir

DATASETS = [
    {
        "csv": os.path.join(CSV_ROOT, "ambel_sta_rosa.csv"),
        "shp_name": "ambel_sta_rosa_ground",
        "dataset_label": "Ambrosia Sta Rosa",
    },
    {
        "csv": os.path.join(CSV_ROOT, "ambel_cato_maiz.csv"),
        "shp_name": "ambel_cato_maiz_ground",
        "dataset_label": "Cato Maiz",
    },
    {
        "csv": os.path.join(CSV_ROOT, "ambel_trigo_corregidas.csv"),
        "shp_name": "ambel_trigo_corregidas_ground",
        "dataset_label": "Trigo Corregidas",
    },
]

os.makedirs(GEO_EXPORTS, exist_ok=True)

all_gdfs = []

for ds in DATASETS:
    csv_path = ds["csv"]
    shp_name = ds["shp_name"]
    label = ds["dataset_label"]

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")

    # Filter rows with valid GPS
    df_gps = df.dropna(subset=["latitude", "longitude"]).copy()
    dropped = len(df) - len(df_gps)
    if dropped > 0:
        print(f"  Dropped {dropped} rows without GPS")

    # Create GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df_gps["longitude"], df_gps["latitude"])]
    gdf = gpd.GeoDataFrame(df_gps, geometry=geometry, crs="EPSG:4326")

    # Save individual shapefile
    shp_path = os.path.join(GEO_EXPORTS, f"{shp_name}.shp")
    gdf.to_file(shp_path)
    print(f"  Saved: {shp_path} ({len(gdf)} features)")

    # Add dataset column for merged file
    gdf["dataset"] = label
    all_gdfs.append(gdf)

# Merge all into one
print(f"\nMerging {len(all_gdfs)} datasets...")
gdf_all = pd.concat(all_gdfs, ignore_index=True)
gdf_all = gpd.GeoDataFrame(gdf_all, geometry="geometry", crs="EPSG:4326")

all_shp = os.path.join(GEO_EXPORTS, "ambel_all_ground.shp")
gdf_all.to_file(all_shp)
print(f"Saved: {all_shp} ({len(gdf_all)} features)")

print(f"\nDone. All shapefiles in: {GEO_EXPORTS}")
print(f"\nSummary:")
for ds in DATASETS:
    shp = os.path.join(GEO_EXPORTS, f"{ds['shp_name']}.shp")
    print(f"  {ds['dataset_label']}: {shp}")
print(f"  ALL MERGED: {all_shp}")
