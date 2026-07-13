from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vision_toolkit.ca_correction.blending import ModeConstraint, SafetyBlender
from vision_toolkit.ca_correction.manifolds import (
    CorrectionInterpolator,
    LogRatioComputer,
    ManifoldBuilder,
)
from vision_toolkit.ca_correction.settings import (
    CASettings,
    MODE_STANDARD,
    load_ca_settings,
)


class ChannelCorrector:

    def __init__(
        self,
        radius: int = None,
        strength: float = None,
        mode: int = None,
        refine: bool = True,
        safety: float = None,
        log_threshold: float = None,
        settings: CASettings = None,
    ):
        settings = settings or load_ca_settings()
        defaults = settings.defaults

        self._radius = radius if radius is not None else defaults.radius
        self._strength = strength if strength is not None else defaults.strength
        self._mode = mode if mode is not None else MODE_STANDARD
        self._safety = safety if safety is not None else defaults.safety
        self._log_threshold = (
            log_threshold if log_threshold is not None else defaults.log_threshold
        )
        self._clip_min = settings.numerical.clip_min
        self._clip_max_mult = settings.numerical.clip_max_multiplier

        self._log_ratio = LogRatioComputer(self._log_threshold, settings=settings)
        self._manifold_builder = ManifoldBuilder(
            self._radius, refine, settings=settings
        )
        self._interpolator = CorrectionInterpolator(settings=settings)
        self._mode_constraint = ModeConstraint(self._mode, settings=settings)
        self._safety_blender = SafetyBlender(self._safety, settings=settings)

    def __call__(self, target: NDArray, guide: NDArray) -> NDArray:
        log_ratio, weight = self._log_ratio(target, guide)
        lower, upper = self._manifold_builder(guide, log_ratio, weight)

        correction = self._interpolator(guide, lower, upper)
        correction = self._mode_constraint(correction)
        correction = correction * self._strength

        corrected = guide * np.power(2.0, correction)

        if self._safety > self._clip_min:
            result = self._safety_blender(target, corrected, guide)
        else:
            result = corrected

        return np.clip(result, self._clip_min, np.max(target) * self._clip_max_mult)

    def __repr__(self) -> str:
        return (
            f"ChannelCorrector(radius={self._radius}, strength={self._strength})"
        )