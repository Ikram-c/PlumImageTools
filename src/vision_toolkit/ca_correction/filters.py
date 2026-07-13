from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

from vision_toolkit.ca_correction.settings import CASettings, load_ca_settings


class BoxFilter:

    def __init__(self, radius: int, settings: CASettings = None):
        settings = settings or load_ca_settings()
        self._radius = radius
        self._size = (
            settings.kernel.size_multiplier * radius + settings.kernel.size_offset
        )

    def __call__(self, img: NDArray) -> NDArray:
        return uniform_filter(
            img.astype(np.float64), size=self._size, mode="reflect"
        )

    def __repr__(self) -> str:
        return f"BoxFilter(radius={self._radius})"


class GuidedFilter:

    def __init__(self, radius: int, eps: float = None, settings: CASettings = None):
        settings = settings or load_ca_settings()
        self._radius = radius
        self._eps = eps if eps is not None else settings.numerical.epsilon_filter
        self._box = BoxFilter(radius, settings)

    def __call__(self, guide: NDArray, src: NDArray) -> NDArray:
        mean_g = self._box(guide)
        mean_s = self._box(src)
        mean_gs = self._box(guide * src)
        mean_gg = self._box(guide * guide)

        cov_gs = mean_gs - mean_g * mean_s
        var_g = mean_gg - mean_g * mean_g

        a = cov_gs / (var_g + self._eps)
        b = mean_s - a * mean_g

        return self._box(a) * guide + self._box(b)

    def __repr__(self) -> str:
        return f"GuidedFilter(radius={self._radius}, eps={self._eps})"