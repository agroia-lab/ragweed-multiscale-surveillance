# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Instituto de Investigaciones Agropecuarias (INIA), Chile
"""
AMBEL Adult SAHI Inference - Full 3-Dataset Run
================================================
Runs SAHI sliced inference on all 3 AMBEL adult plant datasets,
extracts EXIF GPS, and produces per-dataset CSVs.

Datasets:
  1. Ambrosia Sta Rosa  (75 images)
  2. Cato Maiz          (67 images)
  3. Trigo Corregidas   (48 images, 4 subdirs)

Model: run3_lentejas640_b64_2gpu/weights/best.pt (trained weights available
       on request from the corresponding author)

Usage:
    python scripts/03_inference/ambel_sahi_inference_full.py \
        --model weights/best.pt \
        --data-root data/ambel_adult \
        --output-root outputs/ambel_adult_full

Expected data layout under --data-root (field photographs with EXIF GPS,
not distributed with this repository):
    Ambrosia.Sta.Rosa/fotos.proyecto01-02-23/   (75 images)
    CATO_MAIZ/                                  (67 images)
    trigo_corregidas/                           (48 images, 4 subdirs)
"""

import os
import sys
import json
import csv
import glob
import time
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ── Configuration ────────────────────────────────────────────────────

MODEL_PATH = "weights/best.pt"  # run3_lentejas640_b64_2gpu; override with --model
DATA_ROOT = "data/ambel_adult"  # override with --data-root

DATASETS = [
    {
        "name": "ambrosia_sta_rosa",
        "label": "Ambrosia Sta Rosa",
        "path": "Ambrosia.Sta.Rosa/fotos.proyecto01-02-23/",
        "recursive": False,
    },
    {
        "name": "cato_maiz",
        "label": "Cato Maiz",
        "path": "CATO_MAIZ/",
        "recursive": False,
    },
    {
        "name": "trigo_corregidas",
        "label": "Trigo Corregidas",
        "path": "trigo_corregidas/",
        "recursive": True,
    },
]

OUTPUT_ROOT = "outputs/ambel_adult_full"  # override with --output-root
CSV_NAMES = {
    "ambrosia_sta_rosa": "ambel_sta_rosa.csv",
    "cato_maiz": "ambel_cato_maiz.csv",
    "trigo_corregidas": "ambel_trigo_corregidas.csv",
}

SLICE_SIZE = 640
OVERLAP_RATIO = 0.2
CONFIDENCE = 0.25
DEVICE = "cuda:0"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ── EXIF GPS extraction ─────────────────────────────────────────────

def get_gps(image_path):
    """Extract lat, lon, alt from EXIF GPS tags."""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None, None, None
        gps_info = {}
        for tag, value in exif.items():
            if TAGS.get(tag) == "GPSInfo":
                for t in value:
                    gps_info[GPSTAGS.get(t, t)] = value[t]
        if not gps_info:
            return None, None, None

        def to_decimal(dms, ref):
            d, m, s = [float(x) for x in dms]
            dec = d + m / 60 + s / 3600
            if ref in ("S", "W"):
                dec = -dec
            return dec

        lat = to_decimal(
            gps_info.get("GPSLatitude", (0, 0, 0)),
            gps_info.get("GPSLatitudeRef", "N"),
        )
        lon = to_decimal(
            gps_info.get("GPSLongitude", (0, 0, 0)),
            gps_info.get("GPSLongitudeRef", "W"),
        )
        alt = float(gps_info.get("GPSAltitude", 0))
        return lat, lon, alt
    except Exception as e:
        print(f"  GPS error for {os.path.basename(image_path)}: {e}")
        return None, None, None


# ── Collect images ───────────────────────────────────────────────────

def collect_images(dataset):
    """Collect image paths for a dataset."""
    base = os.path.join(DATA_ROOT, dataset["path"])
    if dataset["recursive"]:
        images = []
        for ext in IMAGE_EXTS:
            images.extend(glob.glob(os.path.join(base, "**", f"*{ext}"), recursive=True))
            images.extend(glob.glob(os.path.join(base, "**", f"*{ext.upper()}"), recursive=True))
        # deduplicate
        images = sorted(set(images))
    else:
        images = []
        for f in sorted(os.listdir(base)):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                images.append(os.path.join(base, f))
    return images


# ── Main ─────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="SAHI sliced inference on the AMBEL adult-plant datasets")
    ap.add_argument("--model", default=MODEL_PATH, help="Path to trained YOLO weights (.pt)")
    ap.add_argument("--data-root", default=DATA_ROOT, help="Root folder containing the dataset subfolders")
    ap.add_argument("--output-root", default=OUTPUT_ROOT, help="Output folder for CSVs and summaries")
    ap.add_argument("--device", default=DEVICE, help="Device, e.g. cuda:0 or cpu")
    return ap.parse_args()


def main():
    global MODEL_PATH, DATA_ROOT, OUTPUT_ROOT, DEVICE
    args = parse_args()
    MODEL_PATH, DATA_ROOT, OUTPUT_ROOT, DEVICE = args.model, args.data_root, args.output_root, args.device
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Load model once
    print(f"Loading model: {MODEL_PATH}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE,
        device=DEVICE,
    )
    print("Model loaded.\n")

    all_csvs = {}  # name -> csv_path

    for ds in DATASETS:
        ds_name = ds["name"]
        ds_label = ds["label"]
        ds_outdir = os.path.join(OUTPUT_ROOT, ds_name)
        os.makedirs(ds_outdir, exist_ok=True)

        images = collect_images(ds)
        print(f"{'='*60}")
        print(f"Dataset: {ds_label}")
        print(f"Images:  {len(images)}")
        print(f"Output:  {ds_outdir}")
        print(f"{'='*60}")

        per_image_results = []
        t0 = time.time()

        for i, img_path in enumerate(images):
            img_name = os.path.basename(img_path)
            print(f"  [{i+1}/{len(images)}] {img_name} ... ", end="", flush=True)

            # Run SAHI inference
            result = get_sliced_prediction(
                img_path,
                detection_model,
                slice_height=SLICE_SIZE,
                slice_width=SLICE_SIZE,
                overlap_height_ratio=OVERLAP_RATIO,
                overlap_width_ratio=OVERLAP_RATIO,
            )

            # Count detections by class
            det_by_class = {}
            for pred in result.object_prediction_list:
                cls_name = pred.category.name
                det_by_class[cls_name] = det_by_class.get(cls_name, 0) + 1

            total_det = sum(det_by_class.values())
            nr_ambel = det_by_class.get("AMBEL", 0)

            # Extract GPS
            lat, lon, alt = get_gps(img_path)

            rec = {
                "image": img_name,
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "total_det": total_det,
                "nr_ambel": nr_ambel,
                "det_by_class": det_by_class,
            }
            per_image_results.append(rec)
            print(f"det={total_det}  ambel={nr_ambel}  gps={'OK' if lat else 'NONE'}")

        elapsed = time.time() - t0
        total_dets = sum(r["total_det"] for r in per_image_results)
        total_ambel = sum(r["nr_ambel"] for r in per_image_results)
        gps_count = sum(1 for r in per_image_results if r["latitude"] is not None)

        print(f"\n  Completed {len(images)} images in {elapsed:.1f}s")
        print(f"  Total detections: {total_dets}  |  AMBEL: {total_ambel}")
        print(f"  GPS coverage: {gps_count}/{len(images)}\n")

        # Save detection_summary.json
        summary = {
            "dataset": ds_label,
            "model": MODEL_PATH,
            "slice_size": SLICE_SIZE,
            "overlap": OVERLAP_RATIO,
            "confidence": CONFIDENCE,
            "total_images": len(images),
            "total_detections": total_dets,
            "total_ambel": total_ambel,
            "gps_coverage": f"{gps_count}/{len(images)}",
            "elapsed_seconds": round(elapsed, 1),
            "per_image": [
                {
                    "image": r["image"],
                    "latitude": r["latitude"],
                    "longitude": r["longitude"],
                    "altitude": r["altitude"],
                    "total_det": r["total_det"],
                    "nr_ambel": r["nr_ambel"],
                }
                for r in per_image_results
            ],
        }
        json_path = os.path.join(ds_outdir, "detection_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {json_path}")

        # Save CSV at OUTPUT_ROOT level
        csv_name = CSV_NAMES[ds_name]
        csv_path = os.path.join(OUTPUT_ROOT, csv_name)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "latitude", "longitude", "altitude", "total_det", "nr_ambel"])
            for r in per_image_results:
                writer.writerow([
                    r["image"],
                    r["latitude"] if r["latitude"] is not None else "",
                    r["longitude"] if r["longitude"] is not None else "",
                    r["altitude"] if r["altitude"] is not None else "",
                    r["total_det"],
                    r["nr_ambel"],
                ])
        print(f"  Saved: {csv_path}\n")
        all_csvs[ds_name] = csv_path

    # Final summary
    print("=" * 60)
    print("ALL DATASETS COMPLETE")
    print("=" * 60)
    for name, path in all_csvs.items():
        print(f"  {name}: {path}")
    print(f"\nOutput root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
