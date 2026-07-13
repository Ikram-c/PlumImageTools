from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from vision_toolkit.quality.exposure import ExposureAnalyzer
from vision_toolkit.quality.spatial_frequency import SpatialFrequencyAnalyzer

ANALYSIS_REGISTRY: dict[str, Callable[[np.ndarray, str], Any]] = {}


def register_analysis(name: str):
    def decorator(fn: Callable[[np.ndarray, str], Any]):
        ANALYSIS_REGISTRY[name] = fn
        return fn

    return decorator


@register_analysis("exposure")
def _run_exposure(image: np.ndarray, image_id: str) -> dict[str, Any]:
    return ExposureAnalyzer().analyze_exposure(image, image_id)


@register_analysis("frequency")
def _run_frequency(image: np.ndarray, image_id: str) -> dict[str, Any]:
    return SpatialFrequencyAnalyzer().analyze_frequency(image, image_id)


def analyze_image_quality(image_path: str) -> dict[str, Any]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at path: {image_path}")

    image_id = Path(image_path).name
    results: dict[str, Any] = {}
    for name, analysis_fn in ANALYSIS_REGISTRY.items():
        try:
            results[name] = analysis_fn(image, image_id)
        except Exception as e:
            results[name] = {"error": str(e)}
    return results