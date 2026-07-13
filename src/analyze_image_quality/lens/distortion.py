from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import Optional

import cv2
import imageio.v3 as iio
import lensfunpy

logger = logging.getLogger(__name__)


def _conv_fraction(fraction_string: str) -> Optional[float]:
    negative = 1
    if fraction_string.startswith("-"):
        fraction_string = fraction_string[1:]
        negative = -1
    try:
        return negative * float(
            sum(Fraction(s) for s in fraction_string.split())
        )
    except (ValueError, ZeroDivisionError):
        return None


def lens_fix(
    im_path: str, metadata_dict: dict, output_dir: str
) -> Optional[Path]:
    db = lensfunpy.Database()
    make = metadata_dict.get("Image Make", "")
    model = metadata_dict.get("Image Model", "")

    cams = db.find_cameras(make, model)
    if not cams:
        logger.warning("Camera '%s %s' not found in lensfun database.", make, model)
        return None
    cam = cams[0]

    lens_model = metadata_dict.get("EXIF LensModel")
    lenses = db.find_lenses(cam, lens_model) if lens_model else db.find_lenses(cam)
    if not lenses:
        logger.warning(
            "Lens '%s' not found for camera '%s' in lensfun database.",
            lens_model,
            model,
        )
        return None
    lens = lenses[0]

    focal_str = metadata_dict.get("EXIF FocalLength")
    aperture_str = metadata_dict.get("EXIF ApertureValue")
    if not focal_str or not aperture_str:
        logger.warning("FocalLength or ApertureValue missing in metadata.")
        return None
    focal_length = _conv_fraction(focal_str)
    aperture = _conv_fraction(aperture_str)
    if focal_length is None or aperture is None:
        logger.warning("Could not parse focal length or aperture.")
        return None

    if metadata_dict.get("RelativeAltitude"):
        distance = float(metadata_dict["RelativeAltitude"])
    elif metadata_dict.get("EXIF SubjectDistance"):
        distance = _conv_fraction(metadata_dict["EXIF SubjectDistance"]) or 0
    else:
        distance = 0

    img = iio.imread(im_path)
    h, w = img.shape[:2]

    mod = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
    mod.initialize(focal_length, aperture, distance)
    undistort_map = mod.apply_geometry_distortion()
    img_undistorted = cv2.remap(img, undistort_map, None, cv2.INTER_LANCZOS4)

    base = Path(im_path).stem
    out_path = Path(output_dir) / f"undistorted_{base}.jpg"
    iio.imwrite(out_path, img_undistorted)
    logger.info("Corrected %s -> %s", Path(im_path).name, out_path)
    return out_path