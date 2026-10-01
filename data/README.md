# Data samples

Three small samples with YOLO-format bounding-box labels, for smoke tests and inference demos. They are not meant for training.

| Folder | Source dataset (as named in the preprint) | Images | Classes |
|---|---|---:|---|
| `data/sample/` | Drone, 4 classes (Roboflow `merge-2to11` v5) | 12 | AMBEL, LENCU, POLAV, POLPE |
| `data/sample_cl_seba/` | CL_Seba (640×640, test split) | 12 | AMBEL |
| `data/sample_cl_alberto/` | CL_Alberto (1344×1008, test split) | 8 | AMBEL |

## Drone sample (`data/sample/`)

`data/sample/` contains 12 RGB images (2048×2048 px).

## Provenance

- **Source:** Roboflow export of project `merge-2to11`, version 5, workspace `dlm2lencu`
  (https://universe.roboflow.com/dlm2lencu-o0fxa/merge-2to11/dataset/5), exported 18 October 2024.
- **Licence of the source and of this sample:** CC BY 4.0 (`data/LICENSE`).
- **Preprocessing applied by Roboflow:** auto-orientation (EXIF orientation stripped) and resize to 2048×2048 (stretch). The source export also contains augmented copies; only the images listed below are included here.
- **Classes (EPPO codes), `data/sample/data.yaml`:**

  | id | code | species |
  |---|---|---|
  | 0 | AMBEL | *Ambrosia artemisiifolia* (common ragweed) |
  | 1 | LENCU | *Lens culinaris* (lentil, the crop) |
  | 2 | POLAV | *Polygonum aviculare* |
  | 3 | POLPE | *Polygonum persicaria* |

## Selection criterion

Images were split into three groups (terciles) by their number of AMBEL boxes, and images were drawn from each group with random seed 42. Three difficult cases were added.

| Split in the source | Image (file name prefix) | AMBEL | LENCU | POLAV | POLPE |
|---|---|---:|---:|---:|---:|
| valid | IMG_1469 | 14 | 23 | 1 | 46 |
| valid | IMG_3576 | 3 | 21 | 0 | 5 |
| valid | IMG_3103 | 0 | 23 | 9 | 9 |
| valid | IMG_1429 | 25 | 22 | 1 | 89 |
| valid | IMG_1642 | 19 | 19 | 11 | 0 |
| valid | IMG_1473 | 31 | 19 | 0 | 42 |
| valid | IMG_1496 | 40 | 17 | 0 | 22 |
| valid | IMG_9587 | 1 | 25 | 2 | 313 |
| test | IMG_1075 | 19 | 17 | 0 | 0 |
| test | IMG_1073 | 29 | 18 | 2 | 1 |
| test | IMG_1435 | 35 | 17 | 2 | 43 |
| test | IMG_1428 | 28 | 25 | 0 | 114 |

Box counts are taken from the label files in `data/sample/labels/`.

## Chilean field samples (`data/sample_cl_seba/`, `data/sample_cl_alberto/`)

Images from the two Chilean ragweed datasets used for single-site and multi-domain training in the preprint, both collected by INIA Quilamapu in Santa Rosa fields in 2023 and annotated with a single class (0 = AMBEL, *Ambrosia artemisiifolia*).

- **CL_Seba:** 640×640 px; the full dataset has 1,450 / 411 / 204 images (train / valid / test). The sample has 12 test images.
- **CL_Alberto:** 1344×1008 px, from a different Santa Rosa field; the full dataset has 576 / 55 / 28 images. The sample has 8 test images.
- **Licence:** CC BY 4.0 (`data/LICENSE`).

**Selection criterion (reproducible):** test images were sorted by number of boxes per image and split into terciles; images were drawn from each tercile with random seed 42 (CL_Seba 4 + 4 + 4; CL_Alberto 3 + 3 + 2). Box counts per image: CL_Seba 1–15, CL_Alberto 1–6.

## Privacy

All EXIF/XMP metadata, including any GPS tags, was removed from the copies in this repository (`exiftool -all=`). `exiftool -gps:all -r data examples` returns no GPS tags.

## Layout

```
data/
├── LICENSE          CC BY 4.0 legal code
├── README.md
├── sample/              drone, 4 classes
│   ├── data.yaml        Ultralytics dataset file (path: data/sample)
│   ├── images/          12 × .jpg
│   └── labels/          12 × .txt (YOLO: class cx cy w h, normalised)
├── sample_cl_seba/      CL_Seba, 1 class (data.yaml, 12 images + labels)
└── sample_cl_alberto/   CL_Alberto, 1 class (data.yaml, 8 images + labels)
```

Other datasets are not redistributed. Place them under `data/` as described in the main `README.md` and in the header of each script.
