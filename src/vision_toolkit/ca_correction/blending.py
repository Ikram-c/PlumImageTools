from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vision_toolkit.ca_correction.settings import (
    CASettings,
    MODE_BRIGHTEN_ONLY,
    MODE_DARKEN_ONLY,
    MODE_STANDARD,
    load_ca_settings,
)


class ModeConstraint:

    def __init__(self, mode: int = None, settings: CASettings = None):
        settings = settings or load_ca_settings()
        self._mode = mode if mode is not None else MODE_STANDARD
        clip_min = settings.numerical.clip_min
        self._constraint_map = {
            MODE_STANDARD: lambda x: x,
            MODE_BRIGHTEN_ONLY: lambda x: np.maximum(x, clip_min),
            MODE_DARKEN_ONLY: lambda x: np.minimum(x, clip_min),
        }

    def __call__(self, correction: NDArray) -> NDArray:
        constraint_fn = self._constraint_map.get(self._mode, lambda x: x)
        return constraint_fn(correction)

    def __repr__(self) -> str:
        return f"ModeConstraint(mode={self._mode})"


class SafetyBlender:

    def __init__(
        self,
        safety: float = None,
        threshold: float = None,
        epsilon: float = None,
        settings: CASettings = None,
    ):
        settings = settings or load_ca_settings()
        self._safety = safety if safety is not None else settings.defaults.safety
        self._threshold = (
            threshold if threshold is not None else settings.safety_blender.threshold
        )
        self._epsilon = (
            epsilon if epsilon is not None else settings.numerical.epsilon_small
        )
        self._blend_min = settings.safety_blender.blend_min
        self._blend_max = settings.safety_blender.blend_max
        self._clip_min = settings.numerical.clip_min

    def __call__(
        self, target: NDArray, corrected: NDArray, guide: NDArray
    ) -> NDArray:
        if self._safety <= self._clip_min:
            return corrected
        return self._apply_safety(target, corrected, guide)

    def _apply_safety(
        self, target: NDArray, corrected: NDArray, guide: NDArray
    ) -> NDArray:
        original_ratio = target / (guide + self._epsilon)
        corrected_ratio = corrected / (guide + self._epsilon)
        ratio_change = np.abs(corrected_ratio - original_ratio)
        blend = (
            np.clip(
                (ratio_change - self._threshold) / (self._threshold + self._epsilon),
                self._blend_min,
                self._blend_max,
            )
            * self._safety
        )
        return corrected * (1 - blend) + target * blend

    def __repr__(self) -> str:
        return f"SafetyBlender(safety={self._safety})"