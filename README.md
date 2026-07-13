# vision-toolkit

# vision-toolkit

A Python monorepo consolidating computer vision, geospatial, and dataset tooling into a single package with consistent architectural conventions. Built for marine science imaging pipelines but general-purpose across COCO dataset work, image quality analysis, geometric validation, and video-to-Zarr conversion.

## Overview

`vision-toolkit` merges eleven previously separate codebases into one src-layout package:

| Subpackage | Purpose |
|---|---|
| `coco_tools` | COCO dataset generation, merging, splitting, slicing, validation; sweep-line self-intersection detection; Bloom-filter ID collision handling |
| `coco_tools_overlap` | Nested and overlapping bounding-box detection via coverage-depth sweep line |
| `quality` | Image quality analysis (exposure, spatial frequency), feature detection (SIFT / ORB / AKAZE) |
| `clustering` | Autoencoder-enhanced annotation clustering with detection-optimizer recommendations (anchors, augmentations, class weights) |
| `stats` | Interactive Plotly dashboards for COCO annotation statistics |
| `ca_correction` | Chromatic aberration correction via guided-filter manifolds |
| `geometry` | GJK collision detection (2D Nesterov-accelerated, 3D), support functions, test fixtures |
| `heatmap` | Bounding-box density heatmaps with interactive Plotly controls |
| `registration` | Homography estimation and COCO annotation transformation between image pairs |
| `metadata` | EXIF / XMP / IPTC metadata extraction |
| `lens` | Lens distortion correction via lensfunpy |
| `utils` | Resolution-based image sorting |
| `pipelines` | End-to-end orchestration (lens correction + annotation transformation) |
| `video_zarr` | Parallel video frame extraction to disk or OME-Zarr, with cloud-friendly chunking and part splitting |

## Installation

Requires Python >= 3.10. Managed with [`uv`](https://docs.astral.sh/uv/) and built with hatchling.

```bash
# Base install
uv pip install -e .

# With clustering (PyTorch, scikit-learn)
uv pip install -e ".[clustering]"

# With lens correction (lensfunpy, EXIF stack)
uv pip install -e ".[lens]"

# Everything
uv pip install -e ".[clustering,lens]"
```

The `clustering` and `lens` extras are kept separate deliberately — torch and the lensfun/EXIF stack are heavy dependencies that most of the package does not need.

## Package layout
vision-toolkit/
├── pyproject.toml
├── README.md
├── docs/
│   └── geometry_roadmap.rst
└── src/
└── vision_toolkit/
├── coco_tools.py
├── coco_tools_overlap.py
├── config.yaml
├── quality/
 │   ├── image_utils.py
│   ├── exposure.py
│   ├── spatial_frequency.py
│   ├── feature_matcher.py
│   └── runner.py
├── clustering/
  │   ├── settings.py
│   ├── feature_extraction.py
│   ├── autoencoder.py
│   ├── analyzer.py
│   └── detection_optimizer.py
├── stats/
│   ├── coco_dashboard.py
│   └── gui.py
├── ca_correction/
  │   ├── main.py
│   ├── config.yaml
│   ├── settings.py
        │   ├── filters.py
│   ├── manifolds.py
│   ├── blending.py
│   ├── corrector.py
│   ├── pipeline.py
│   └── api.py
   ├── geometry/
│   ├── main.py
│   ├── config.yaml
        │   ├── settings.py
│   ├── supports.py
│   ├── gjk_2d.py
│   ├── gjk_3d.py
    │   └── fixtures.py
├── heatmap/
│   ├── config.yaml
     │   └── density.py
├── registration/
│   └── annotation_transformer.py
├── metadata/
│   └── extractor.py
       ├── lens/
│   └── distortion.py
├── utils/
 │   └── resolution_grouper.py
├── pipelines/
│   └── lens_correction_pipeline.py
└── video_zarr/
├── main.py
            ├── defaults.yaml
  ├── defaults.py
├── exceptions.py
├── models.py
            ├── capture.py
├── io.py
├── transforms.py
            ├── timestamps.py
  ├── workers.py
├── zarr_writer.py
└── extract.py

## Command-line entry points

Two console scripts are installed:

```bash
# Video frame extraction / OME-Zarr conversion
video-zarr config.yaml [--mode auto|frames|zarr] [--workers N] [--verbose]

# Lens correction + annotation transformation
lens-correct <img_dir> <coco_json> <output_dir> [--detector sift|orb|akaze_opencv]
```

Module-level equivalents also work:

```bash
python -m vision_toolkit.video_zarr config.yaml
python -m vision_toolkit.coco_tools config.yaml split
python -m vision_toolkit.coco_tools config.yaml preprocess
python -m vision_toolkit.ca_correction      # synthetic-image demo
python -m vision_toolkit.geometry           # GJK 2D/3D demos
```

### video-zarr exit codes

Designed for batch/scheduler use (e.g. Nextflow):

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Config error (missing file, invalid YAML, zero video paths) |
| 2 | Extraction error (sampling exceeds duration, unreadable video) |
| 3 | Ran but wrote zero frames (includes the all-outputs-exist skip path) |

`--mode auto` (the default) selects `zarr` when the config YAML contains a `zarr` section, otherwise `frames`.

## Usage examples

### COCO tooling

```python
from vision_toolkit.coco_tools import (
    is_self_intersecting,
    COCODatasetExtender,
    CocoResolutionSplitter,
    validate_coco_ids,
)

# Sweep-line self-intersection check on a flat coordinate list
bowtie = [10.0, 10.0, 90.0, 90.0, 90.0, 10.0, 10.0, 90.0]
assert is_self_intersecting(bowtie)

# Merge two COCO files with Bloom-filtered ID collision remapping
extender = COCODatasetExtender("base.json")
stats = extender.merge_and_save("new.json", "merged.json")

# Split a COCO file by image resolution
splitter = CocoResolutionSplitter.from_config("config.yaml")
resolution_file_map = splitter.run()
```

### Nested / overlapping annotation detection

```python
from vision_toolkit.coco_tools_overlap import find_all_invalid_inds

overlap_invalid, nested, all_invalid = find_all_invalid_inds(rectangles, areas)
```

Handles the "bowl of eggs" case — annotations fully contained within other annotations are reported separately from partial overlaps, and overlap groups are resolved transitively with the largest-area member kept.

### Image quality

```python
from vision_toolkit.quality.runner import analyze_image_quality

results = analyze_image_quality("frame_00000042.tiff")
# results["exposure"], results["frequency"], per-module error isolation
```

New analyses register with a decorator and are picked up automatically:

```python
from vision_toolkit.quality.runner import register_analysis

@register_analysis("texture")
def _run_texture(image, image_id):
    ...
```

### Feature detection

```python
from vision_toolkit.quality.feature_matcher import ImageFeatureMatcher

matcher = ImageFeatureMatcher(img, matcher="sift", image_name="dive_042")
features = matcher.detect()
matcher.analyze_and_save_all_data("output/")
```

Default detector is scikit-image SIFT (BSD-licensed, no OpenCV dependency on this path). ORB is available as a lighter option; OpenCV AKAZE remains registered as `"akaze_opencv"` behind a lazy import for reproducing legacy keypoint CSVs. All detectors normalize to a common `DetectedFeatures` shape with `(x, y)` coordinates and degree angles matching the historical CSV schema. CSV outputs include a `detector` column and detector-tagged filenames.

### Annotation clustering

```python
from vision_toolkit.clustering.analyzer import run_clustering_analysis
from vision_toolkit.clustering.settings import AutoencoderSettings

analyzer, results = run_clustering_analysis(
    "annotations.json",
    output_dir="clustering_results",
    autoencoder_settings=AutoencoderSettings(architecture="variational", epochs=80),
)
```

Architectures: `shallow`, `deep`, `sparse`, `variational`. Random Forest feature-importance parameters adapt to discovered cluster count. Detection recommendations:

```python
from vision_toolkit.clustering.detection_optimizer import (
    summarise_optimizer_recommendations,
)

recs = summarise_optimizer_recommendations(analyzer.best_result, feature_df)
# anchor boxes, augmentation policy, class weighting advice
```

### Chromatic aberration correction

```python
from vision_toolkit.ca_correction.api import correct_chromatic_aberration

corrected = correct_chromatic_aberration(image, radius=10, strength=0.8, guide="green")
```

### Geometry

```python
import numpy as np
from vision_toolkit.geometry.gjk_2d import gjk_nesterov_accelerated_2d_intersection
from vision_toolkit.geometry.supports import coco_bbox_support, coco_segmentation_support

a = coco_bbox_support([10, 10, 50, 30])
b = coco_segmentation_support(annotation["segmentation"][0])
overlapping = gjk_nesterov_accelerated_2d_intersection(a, b)
```

Test fixtures live in `geometry.fixtures`: the self-intersecting "eight" polygon, seeded separated shapes, and nested/mixed rectangle generators with known ground truth for asserting the sweep-line and nesting detectors.

### Bounding-box density heatmap

```python
from pathlib import Path
from vision_toolkit.heatmap.density import HeatmapApplication

app = HeatmapApplication(output_dir=Path("output"))
html_path = app.run(Path("instances_val2017.json"))  # show=True to open browser
```

### Lens correction pipeline

```bash
lens-correct ./drone_images ./annotations.json ./output --detector sift
```

Stage 1 extracts EXIF/XMP metadata per image and applies lensfunpy distortion correction into `output/corrected/`. Stage 2 computes per-image homographies between original and corrected images (RANSAC over cross-checked descriptor matches) and rewrites all bboxes, segmentations, and keypoints into `output/annotations_corrected.json`, dropping degenerate results and reporting stats. Images without a lensfun database match are skipped consistently in both stages.

### Video to OME-Zarr

Config-driven. Example YAML:

```yaml
video_paths: ./survey_videos        # file, list, or directory (scanned by extension)
frames_dir: ./frames
colour_space: GRAYSCALE             # or BGR
norm:
  enabled: true
  pmin: 1.0
  pmax: 99.8
every: 5
chunk_size: 1000

sample:                             # optional temporal window
  offset_seconds: 30.0
  duration_seconds: 120.0

crop:                               # optional spatial crop
  x0: 0
  y0: 100
  x1: 1920
  y1: 980

zarr:                               # presence selects zarr mode under --mode auto
  store_path: ./zarr_out
  chunk_frames: 64
  compressor: default               # blosc-zstd L3 bitshuffle
  max_file_size_mb: 500             # 0 = single store; >0 splits into parts
  zip_store: false
```

```bash
video-zarr config.yaml
```

Frames mode fans chunks out over a multiprocessing pool with a shared progress counter. Zarr mode streams single-process into OME-NGFF v0.4 stores with per-frame timestamps, part splitting by estimated compressed size, and optional zip packaging. Video extension matching is case-insensitive (`.MP4` works); frame discovery matches suffixes case-insensitively (`.TIFF` == `.tiff`). TIFF reading routes through `tifffile` and supports both chunky (`PlanarConfiguration=1`) and planar (`PlanarConfiguration=2`) files, converting to cv2-convention BGR arrays regardless of storage layout.

## Conventions

All code in this repository follows a consistent house style:

- **src layout** with hatchling builds, managed via `uv`
- **Frozen dataclasses** for all configuration and value objects
- **YAML-driven settings** hydrated into typed dataclasses via a shared `hydrate` helper — no dict-backed config objects, no import-time singletons
- **Registry patterns** for extensible components (feature detectors, quality analyses, dashboard views)
- **Dependency injection** — settings objects are constructor parameters with sensible defaults, never module globals
- **Logging over print** throughout; CLIs configure logging at the entry point
- **PEP 8**, no inline comments, sparse docstrings
- Bounded loops and explicit validation in the spirit of the NASA Power of 10 rules

## Testing hooks

The package ships its own ground-truth generators, so the core algorithms are testable without real data:

- `coco_tools.CocoDatasetGenerator` — synthetic COCO datasets with known polygon annotations
- `geometry.fixtures` — self-intersecting polygons, separated shapes, nested rectangles with expected indices
- `clustering` — operates on any COCO JSON, including generated ones

Example assertions:

```python
from vision_toolkit.coco_tools import is_self_intersecting
from vision_toolkit.coco_tools_overlap import find_overlapping
from vision_toolkit.geometry.fixtures import (
    self_intersecting_eight,
    generate_nested_rectangles,
)

fixture = self_intersecting_eight()
assert is_self_intersecting(fixture.points.flatten().tolist())

rects, expected_nested, _ = generate_nested_rectangles(seed=7)
_, nested = find_overlapping(rects)
assert nested == expected_nested
```

## Known gaps / roadmap

Tracked in `docs/geometry_roadmap.rst`. Outstanding:

- **Graham scan module** — referenced by the roadmap's self-intersection path; source file pending
- **EFD contour model** — needed for the connected-components boundary re-definition stage
- **`TextureAnalyzer` / `LaplacianVariance`** — quality analyses referenced by the original runner; will register via `@register_analysis` when added
- **`video_zarr.capture.FrameIterator` / `ChunkHeap`** — currently exported but unused; retained pending a decision on the resumable-extraction feature
- **Zarr streaming for `PreProcess`** — annotation loading still goes through pycocotools rather than a Zarr-backed reader
- **Multi-page TIFF policy** — the TIFF reader takes the first IFD only, matching prior cv2 behaviour
- Full-dataset zarr skip path returns 0 frames written, which maps to exit code 3; revisit if scheduler retry semantics need "cached" distinguished differently

## Behavioural changes from the original scripts

Consolidation fixed several latent bugs; if you are migrating from the standalone scripts, these are the diffs that can change output:

1. **Slice generation** — resolutions are now paired with *their own* overlap ratios via a dict; the old zipped-sorted-sets approach could silently misalign them.
2. **`CorrectionPipeline`** — processes all stages; the old version hard-coded exactly three heap pops.
3. **Homography matching** — descriptor matching is norm-aware (Hamming for binary, L2 for float); the old `cv2.NORM_HAMMING` hard-code was wrong for SIFT. The adaptive distance-threshold heuristic was removed in favour of cross-check + RANSAC.
4. **Deep autoencoder** — hidden layer dimension is floored to `int`; the original computed a float and crashed `nn.Linear`.
5. **`SafetyBlender`** — scalar skip condition is an early return; the old code built and discarded a full-size `np.where` array.
6. **Shape fixtures** — single `np.random.default_rng` seeding replaces mixed `random`/`np.random` globals, so exact shapes for a given seed differ from the old plotting script (properties are preserved).
7. **Frame path discovery** — exact case-insensitive suffix match replaces substring glob (`not_a_tiff` no longer matches `*tiff`).
8. **`lens_fix` output naming** — the `undistorted_` prefix is stripped by the pipeline inside the isolated `corrected/` directory so annotation matching works by basename.
9. **`HeatmapApplication.run`** — no longer opens a browser by default; pass `show=True`.

## License

Internal tooling. GJK 3D implementation derived from [vurtun's gist](https://gist.github.com/vurtun/29727217c269a2fbf4c0ed9a1d11cb40); 2D Nesterov-accelerated GJK based on [distance3d](https://github.com/AlexanderFabisch/distance3d).