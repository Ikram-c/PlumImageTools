from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vision_toolkit.ca_correction.filters import GuidedFilter
from vision_toolkit.ca_correction.settings import CASettings, load_ca_settings


class LogRatioComputer:

    def __init__(
        self,
        threshold: float = None,
        epsilon: float = None,
        settings: CASettings = None,
    ):
        settings = settings or load_ca_settings()
        self._threshold = (
            threshold if threshold is not None else settings.defaults.log_threshold
        )
        self._epsilon = (
            epsilon if epsilon is not None else settings.numerical.epsilon_small
        )

    def __call__(
        self, target: NDArray, guide: NDArray
    ) -> tuple[NDArray, NDArray]:
        target_safe = np.maximum(target, self._epsilon)
        guide_safe = np.maximum(guide, self._epsilon)
        log_ratio = np.log2(target_safe) - np.log2(guide_safe)

        abs_ratio = np.abs(log_ratio)
        high_diff_mask = abs_ratio > self._threshold

        weight = np.where(
            high_diff_mask, self._threshold / abs_ratio, np.ones_like(log_ratio)
        )
        clamped = np.clip(log_ratio, -self._threshold, self._threshold)
        weighted_ratio = np.where(high_diff_mask, clamped, log_ratio)
        return weighted_ratio, weight

    def __repr__(self) -> str:
        return f"LogRatioComputer(threshold={self._threshold})"


class ManifoldBuilder:

    def __init__(self, radius: int, refine: bool = True, settings: CASettings = None):
        settings = settings or load_ca_settings()
        self._radius = radius
        self._refine = refine
        self._guided_filter = GuidedFilter(radius, settings=settings)
        self._epsilon = settings.numerical.epsilon_tiny

    def __call__(
        self, guide: NDArray, log_ratio: NDArray, weight: NDArray
    ) -> tuple[NDArray, NDArray]:
        min_val, max_val = np.min(guide), np.max(guide)
        range_val = max_val - min_val + self._epsilon
        normalized_guide = (guide - min_val) / range_val

        weighted_ratio = log_ratio * weight
        lower_weight = (1.0 - normalized_guide) * weight
        upper_weight = normalized_guide * weight

        lower = self._filtered_manifold(guide, weighted_ratio, lower_weight)
        upper = self._filtered_manifold(guide, weighted_ratio, upper_weight)

        if self._refine:
            lower = self._guided_filter(guide, lower)
            upper = self._guided_filter(guide, upper)
        return lower, upper

    def _filtered_manifold(
        self, guide: NDArray, weighted_ratio: NDArray, weight: NDArray
    ) -> NDArray:
        num = self._guided_filter(guide, weighted_ratio * weight)
        den = self._guided_filter(guide, weight)
        return num / (den + self._epsilon)

    def __repr__(self) -> str:
        return f"ManifoldBuilder(radius={self._radius}, refine={self._refine})"


class CorrectionInterpolator:

    def __init__(self, epsilon: float = None, settings: CASettings = None):
        settings = settings or load_ca_settings()
        self._epsilon = (
            epsilon if epsilon is not None else settings.numerical.epsilon_tiny
        )
        self._clip_min = settings.numerical.clip_min
        self._normalize_threshold = settings.numerical.normalize_threshold

    def __call__(
        self, guide: NDArray, lower_manifold: NDArray, upper_manifold: NDArray
    ) -> NDArray:
        min_val, max_val = np.min(guide), np.max(guide)
        range_val = max_val - min_val + self._epsilon
        t = np.clip(
            (guide - min_val) / range_val, self._clip_min, self._normalize_threshold
        )
        return (1.0 - t) * lower_manifold + t * upper_manifold

    def __repr__(self) -> str:
        return "CorrectionInterpolator()"