from __future__ import annotations

import cv2
import numpy as np
from csbdeep.utils import normalize


def validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if image.ndim not in (2, 3):
        raise ValueError("Image must be 2D (grayscale) or 3D (color).")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    validate_image(image)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def to_normalized_grayscale(
    image: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.8,
) -> np.ndarray:
    gray = to_grayscale(image)
    norm_img = normalize(gray, percentile_low, percentile_high, axis=(0, 1))
    return (norm_img * 255).astype(np.uint8)