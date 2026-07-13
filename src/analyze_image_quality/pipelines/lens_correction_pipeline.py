from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any, Generator

from vision_toolkit.lens.distortion import lens_fix
from vision_toolkit.metadata.extractor import MetadataExtractor
from vision_toolkit.registration.annotation_transformer import (
    AnnotationTransformer,
    HomographyEstimator,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
)


def _image_files(directory: Path) -> Generator[Path, None, None]:
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _strip_prefix(corrected_dir: Path) -> None:
    for path in corrected_dir.glob("undistorted_*"):
        target = corrected_dir / path.name.removeprefix("undistorted_")
        if not target.exists():
            shutil.move(str(path), str(target))


def correct_images(img_dir: Path, corrected_dir: Path) -> dict[str, int]:
    corrected_dir.mkdir(parents=True, exist_ok=True)
    counts = {"corrected": 0, "skipped": 0, "failed": 0}

    for im_path in _image_files(img_dir):
        try:
            metadata = MetadataExtractor(str(im_path)).run_metadata_extractor()
        except (OSError, ValueError) as e:
            logger.warning("Metadata failed for %s: %s", im_path.name, e)
            counts["failed"] += 1
            continue

        result = lens_fix(str(im_path), metadata, str(corrected_dir))
        if result is None:
            counts["skipped"] += 1
        else:
            counts["corrected"] += 1

    _strip_prefix(corrected_dir)
    return counts


def run(
    img_dir: Path,
    coco_json: Path,
    output_dir: Path,
    detector: str = "sift",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir = output_dir / "corrected"

    logger.info("Stage 1/2: lens correction from %s", img_dir)
    correction_counts = correct_images(img_dir, corrected_dir)
    logger.info(
        "Lens correction: %d corrected, %d skipped (no lensfun match), "
        "%d failed",
        correction_counts["corrected"],
        correction_counts["skipped"],
        correction_counts["failed"],
    )

    logger.info("Stage 2/2: annotation transformation")
    transformer = AnnotationTransformer(
        coco_json_path=str(coco_json),
        orig_img_dir=str(img_dir),
        corrected_img_dir=str(corrected_dir),
        estimator=HomographyEstimator(detector=detector),
    )
    stats = transformer.transform_dataset_annotations(
        str(output_dir / "annotations_corrected.json")
    )
    return {"lens_correction": correction_counts, "transformation": stats}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Lens-correct an image set and transform its COCO annotations.",
    )
    parser.add_argument("img_dir", type=Path)
    parser.add_argument("coco_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--detector", default="sift", choices=["sift", "orb", "akaze_opencv"])
    args = parser.parse_args()

    if not args.img_dir.is_dir():
        raise SystemExit(f"Image directory not found: {args.img_dir}")
    if not args.coco_json.is_file():
        raise SystemExit(f"COCO JSON not found: {args.coco_json}")

    results = run(args.img_dir, args.coco_json, args.output_dir, args.detector)
    logger.info("Pipeline complete: %s", results)


if __name__ == "__main__":
    main()