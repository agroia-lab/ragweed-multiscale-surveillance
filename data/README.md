# Data sample

`data/sample/` contains 12 RGB images (2048×2048 px) with YOLO-format bounding-box labels, for smoke tests and inference demos. It is not meant for training.

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

## Privacy

All EXIF/XMP metadata, including any GPS tags, was removed from the copies in this repository (`exiftool -all=`). `exiftool -gps:all -r data examples` returns no GPS tags.

## Layout

```
data/
├── LICENSE          CC BY 4.0 legal code
├── README.md
└── sample/
    ├── data.yaml    Ultralytics dataset file (path: data/sample)
    ├── images/      12 × .jpg
    └── labels/      12 × .txt (YOLO: class cx cy w h, normalised)
```

Other datasets are not redistributed. Place them under `data/` as described in the main `README.md` and in the header of each script.
