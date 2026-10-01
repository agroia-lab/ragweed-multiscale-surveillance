# From Plant Detection to Satellite Mapping: A Multi-Scale AI Toolkit for Ragweed Surveillance under Climate Change

Code and a small data sample for the preprint:

> León Gutiérrez LF, Ramírez C, Henríquez A, Contreras S (2026). *From Plant Detection to Satellite Mapping: A Multi-Scale AI Toolkit for Ragweed Surveillance under Climate Change.* bioRxiv. https://doi.org/10.64898/2026.09.28.755009

Instituto de Investigaciones Agropecuarias (INIA), Chillán, Chile.
Corresponding author: Lorenzo F. León Gutiérrez (lleon@inia.cl).

## Abstract

Climate change is reshaping weed population dynamics, creating non-stationary management challenges for which static decision rules are insufficient. This chapter presents and evaluates an open surveillance toolkit of artificial intelligence tools for recognizing and mapping *Ambrosia artemisiifolia* (common ragweed) at complementary spatial scales, tested across two phenological phases (seedling, September 2024; adult, December 2024) in a lentil paddock in central Chile. At the field scale, a YOLOv11 detection model coupled with Slicing Aided Hyper Inference achieved mAP50 of 0.886. Cross-domain experiments revealed that single-site models collapse catastrophically when deployed in new geographies (mAP50 dropping to 0.108), but multi-domain training recovers performance to 0.874. At the satellite scale, PRESTO foundation model embeddings correlated strongly with ground-truth weed density (r = 0.739); geographically weighted regression explained up to 91% of local density variance. All methods are released as the ragweed-ai-toolkit (https://github.com/agroia-lab/ragweed-ai-toolkit), a modular open-source Python package enabling adaptation to new species and geographies.

## What this repository contains

This repository holds the **original scripts that produced the results reported in the preprint**, organised by workflow stage, plus small annotated image samples (32 images) for smoke tests.

**How it relates to [`agroia-lab/ragweed-ai-toolkit`](https://github.com/agroia-lab/ragweed-ai-toolkit).** That repository (MIT) is a February 2026 refactor of these methods into an installable Python package. This repository is the analysis code as it was run for the preprint. Paths were turned into command-line arguments or repository-relative defaults; the scientific logic was not changed. Use the toolkit to adapt the methods to new species or sites, and use this repository to inspect or reproduce the preprint's analysis.

## Why AGPL-3.0

Detection and training use [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0. These scripts import and extend it, so they are released under the same licence: **GNU Affero General Public License v3.0 or later** (`LICENSE`). If you modify the code and offer it to others, including as a network service, you must make your source available under the same terms.

## Multi-scale workflow

```
 01_data           02_training        03_inference           04_evaluation
 build YOLO   -->  YOLOv11 training -> SAHI sliced inference -> cross-domain mAP
 datasets          (single / multi-    + EXIF GPS -> GeoPackage  per source database
 (CL + intl.)      domain)             / shapefiles
                                            |
                                            v
 05_domain_shift                        06_satellite
 orthomosaic tiling -> ResNet50     -->  Sentinel-1/2 PRESTO embeddings (openEO)
 embeddings -> MMD vs reference          + Sentinel-2 indices vs weed density (kriging)
 databases -> detections to GPKG         -> bivariate Moran's I / LISA -> GWR
```

Run every command from the repository root.

### 01 — Data (`scripts/01_data/`)

| Script / config | Purpose |
|---|---|
| `build_international_dataset.py` | Unifies four public ragweed databases into one single-class (AMBEL) YOLO set with a `manifest.csv` of source per image |
| `build_combined_dataset.py` | Merges the Chilean sets (CL_Seba, CL_Alberto) with the international set using source-stratified splits |
| `configs/international_databases.yaml` | Source databases, formats and class remapping |
| `configs/data_ambel-*.yaml` | Ultralytics dataset files used for training and evaluation |

```bash
python scripts/01_data/build_international_dataset.py \
    --config scripts/01_data/configs/international_databases.yaml \
    --output data/ambel-international-v1 --dry-run
python scripts/01_data/build_combined_dataset.py \
    --cl-seba data/ambrosia-lentejas-seba-v1 \
    --cl-alberto data/ambrosia.dataset_alberto/ambrosia.dataset \
    --international data/ambel-international-v1 \
    --output data/ambel-combined-intl-v1 --seed 42
```

### 02 — Training (`scripts/02_training/`)

```bash
yolo settings datasets_dir=$(pwd)   # relative `path:` entries resolve from the repo root
python scripts/02_training/train_yolo_cli.py \
    --model yolo11m.pt --data scripts/01_data/configs/data_ambel-lentejas-seba-v1.yaml \
    --epochs 50 --imgsz 640 --batch 16 --device 0 --project runs/train --name ambel
```

### 03 — Inference and georeferencing (`scripts/03_inference/`)

```bash
# SAHI sliced vs direct inference on the included sample (model weights: see below)
python scripts/03_inference/sahi_inference_cli.py \
    --model weights/best.pt --images data/sample/images --labels data/sample/labels \
    --slice 1024 --output outputs/sahi_comparison

# Adult-plant field photographs: SAHI + EXIF GPS -> CSV, then shapefiles
python scripts/03_inference/ambel_sahi_inference_full.py \
    --model weights/best.pt --data-root data/ambel_adult --output-root outputs/ambel_adult_full
python scripts/03_inference/ambel_create_shapefiles.py \
    --csv-root outputs/ambel_adult_full --output-dir outputs/geo_exports

# Per-photo AMBEL counts + GPS -> GeoPackage
python scripts/03_inference/export_ambel_geo.py \
    --images data/ground_photos --summary outputs/sahi/summary.json \
    --output-name ground_ambel --output-dir outputs/geo_exports --device-type ground
```

`scripts/utils/paths.py` and `scripts/utils/geo_data_utils.py` are helper modules imported by `export_ambel_geo.py`.

### 04 — Cross-domain evaluation (`scripts/04_evaluation/`)

```bash
python scripts/04_evaluation/crossdomain_eval.py \
    --model weights/best.pt \
    --data scripts/01_data/configs/data_ambel-international-v1-test.yaml \
    --manifest data/ambel-international-v1/manifest.csv \
    --output outputs/ambel_crossdomain/eval_cl_on_international
python scripts/04_evaluation/crossdomain_compare.py \
    --evals outputs/ambel_crossdomain/*/crossdomain_metrics.json \
    --output outputs/ambel_crossdomain/comparison --baseline-map50 0.886
```

### 05 — Domain shift and orthomosaic mapping (`scripts/05_domain_shift/`)

```bash
python scripts/05_domain_shift/tile_orthomosaic.py \
    --raster data/ortho/orthomosaic.tif --output outputs/ortho_tiles_1024 --tile-size 1024
python scripts/05_domain_shift/embed_ortho_tiles.py \
    --tiles outputs/ortho_tiles_1024 --output outputs/ortho_embeddings --model resnet50
python scripts/05_domain_shift/compute_ortho_domain_shift.py \
    --tile-embeddings outputs/ortho_embeddings/embeddings.npz \
    --reference-embeddings data/international_databases/ragweed_embeddings/embeddings.npz \
    --output outputs/domain_shift_report
python scripts/05_domain_shift/detections_to_geopackage.py \
    --detections outputs/sahi_ortho/detections.json \
    --manifest outputs/ortho_tiles_1024/tile_manifest.csv \
    --output outputs/ambel_ortho_detections.gpkg --crs EPSG:32719
python scripts/05_domain_shift/generate_figures.py \
    --data-dir data/international_databases/ragweed_embeddings --out-dir outputs/figures
```

`embedding_backbone.py` holds the feature extractor and dimensionality-reduction functions that `embed_ortho_tiles.py` imports, copied verbatim from the original embedding module. `generate_figures.py` expects pre-computed `mmd_matrix.csv`, `generalization_risk_scores.csv` and `embeddings_reduced.csv` for the nine image collections.

### 06 — Satellite scale (`scripts/06_satellite/`)

```bash
# PRESTO embeddings (Sentinel-1/2, Jul-Dec 2024) via openEO / Copernicus Data Space,
# aligned with the kriged weed-density surfaces
python scripts/06_satellite/extract_presto_lencu_paddock.py \
    --paddock-kml "data/satellite/lencu paddock.kml" --smartmap-dir data/satellite/smartmap_outputs
# Sentinel-2 spectral indices vs weed density
python scripts/06_satellite/analyze_sentinel_weed_density.py \
    --sentinel-tiff "data/satellite/2024-09-24, Boundary1.data.tif" \
    --smartmap-dir data/satellite/smartmap_outputs
# Spatial statistics on the aligned table
python scripts/06_satellite/spatial_correlation_lencu.py --data outputs/lencu_presto/analysis_results.csv
python scripts/06_satellite/gwr_lencu.py --data outputs/lencu_presto/analysis_results.csv
```

`extract_presto_embeddings.py` provides the openEO/WorldCereal extraction functions used by `extract_presto_lencu_paddock.py`. It needs a separate environment with the [WorldCereal classification](https://github.com/WorldCereal/worldcereal-classification) package and a Copernicus Data Space account (`--authenticate`).

## Installation

```bash
conda env create -f environment.yml
conda activate ragweed-multiscale
# or: python -m pip install -r requirements.txt   (Python 3.10)
```

Versions marked `# version not pinned in the original run` in `requirements.txt` were not recorded for the runs reported in the preprint.

## Data

**Included (CC BY 4.0, EXIF metadata removed; provenance and selection criteria in `data/README.md`):**

- `data/sample/`: twelve 2048×2048 drone images with YOLO labels for four classes (AMBEL, LENCU, POLAV, POLPE), from a public Roboflow export.
- `data/sample_cl_seba/`: twelve test images from CL_Seba (single class, AMBEL).
- `data/sample_cl_alberto/`: eight test images from CL_Alberto (single class, AMBEL).

**Included: `examples/106_DJI_0389_comparison.jpg`.** A drone image from the Santa Rosa lentil paddock (left) next to the SAHI detections on it (right; slice 2048, 1,570 detections). EXIF metadata was removed.

**Not redistributed.** The public datasets used for cross-domain experiments must be downloaded from their original sources, and cited as each source requests:

- **3SeasonWeedDet10** (Michigan, 2021–2023): Zenodo record 14861516, https://zenodo.org/records/14861516
- **Weed-crop dataset in precision agriculture** (Mendeley Data): https://doi.org/10.17632/mthv4ppwyw.2
- **ND Individual** and **ND Aerial** (ImageWeeds, North Dakota), **Purdue 4Weed**, **WeedCube (USDA)**: see original sources.

Place them under `data/international_databases/` following the layout in `scripts/01_data/configs/international_databases.yaml`. The Chilean training sets, field photographs, orthomosaics, Sentinel clips and kriging surfaces are not distributed. Each script documents the inputs it expects in its header.

**Kriging.** Weed-density surfaces were interpolated by ordinary kriging in SmartMap (CENIA, Chile). That software is not part of this repository. The satellite scripts read its exported grids (`1_Krig_<SPECIES>_Grid_Map.tiff`, `0_Dados.csv`).

**Trained weights.** The trained model weights are available from the corresponding author on request.

## Citation

```bibtex
@article{leongutierrez2026ragweed,
  title   = {From Plant Detection to Satellite Mapping: A Multi-Scale AI Toolkit for Ragweed Surveillance under Climate Change},
  author  = {Le{\'o}n Guti{\'e}rrez, Lorenzo F. and Ram{\'i}rez, Cristofer and Henr{\'i}quez, Alejandra and Contreras, Sebasti{\'a}n},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.28.755009},
  url     = {https://www.biorxiv.org/content/10.64898/2026.09.28.755009v1}
}
```

See also `CITATION.cff`.

## Acknowledgements

This work was funded by the Fundación para la Innovación Agraria (FIA) and the Subsecretaría de Agricultura, Chile.

## License

- **Code:** GNU AGPL-3.0-or-later (`LICENSE`). Copyright (C) 2026 Instituto de Investigaciones Agropecuarias (INIA), Chile.
- **Data samples** (`data/sample*/`): CC BY 4.0 (`data/LICENSE`).
- **Example panel** (`examples/`): CC BY 4.0, like the data sample.

## Contact

Lorenzo F. León Gutiérrez, INIA, lleon@inia.cl
