from __future__ import annotations

import numpy as np

from vision_toolkit.ca_correction.pipeline import CACorrectRGB
from vision_toolkit.ca_correction.settings import GREEN


def create_synthetic_test_image(
    height: int = 256,
    width: int = 256,
    gaussian_sigma: float = 50.0,
    red_offset: float = 3.0,
    green_offset: float = 0.0,
    blue_offset: float = -3.0,
    intensity_scale: float = 255.0,
    random_seed: int = 42,
) -> np.ndarray:
    np.random.seed(random_seed)
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    def channel_fn(offset: float) -> np.ndarray:
        return np.exp(-((r - offset) ** 2) / (2 * gaussian_sigma**2))

    synthetic = np.stack(
        [channel_fn(red_offset), channel_fn(green_offset), channel_fn(blue_offset)],
        axis=2,
    )
    return (synthetic * intensity_scale).astype(np.float64)


def run_demo() -> tuple[np.ndarray, np.ndarray]:
    synthetic = create_synthetic_test_image()
    corrector = CACorrectRGB(radius=10, strength=0.8, guide=GREEN)
    corrected = corrector(synthetic)

    print(repr(corrector))
    print(str(corrector))
    print(f"Filter size: {len(corrector)}")
    print(f"Active: {bool(corrector)}")
    print(f"Radius setting: {corrector['radius']}")
    print(f"Has strength: {'strength' in corrector}")
    return synthetic, corrected


if __name__ == "__main__":
    run_demo()