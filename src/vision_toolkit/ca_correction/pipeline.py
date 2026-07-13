from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray

from vision_toolkit.ca_correction.corrector import ChannelCorrector
from vision_toolkit.ca_correction.settings import (
    BLUE,
    CASettings,
    CHANNEL_COUNT,
    GREEN,
    MODE_STANDARD,
    RED,
    load_ca_settings,
)


class CACorrectRGB:

    def __init__(
        self,
        radius: int = None,
        strength: float = None,
        guide: int = None,
        mode: int = None,
        refine_manifolds: bool = True,
        safety: float = None,
        log_threshold: float = None,
        settings: CASettings = None,
    ):
        settings = settings or load_ca_settings()
        defaults = settings.defaults

        self._settings = settings
        self._radius = radius if radius is not None else defaults.radius
        self._strength = strength if strength is not None else defaults.strength
        self._guide = guide if guide is not None else GREEN
        self._mode = mode if mode is not None else MODE_STANDARD
        self._refine = refine_manifolds
        self._safety = safety if safety is not None else defaults.safety
        self._log_threshold = (
            log_threshold if log_threshold is not None else defaults.log_threshold
        )

        self._corrector = ChannelCorrector(
            self._radius,
            self._strength,
            self._mode,
            refine_manifolds,
            self._safety,
            self._log_threshold,
            settings=settings,
        )

    def __call__(self, image: NDArray) -> NDArray:
        validated = self._validate_input(image)
        normalized, scale = self._normalize(validated)

        result = normalized.copy()
        guide_channel = normalized[:, :, self._guide]

        for ch in range(CHANNEL_COUNT):
            if ch == self._guide:
                continue
            result[:, :, ch] = self._corrector(normalized[:, :, ch], guide_channel)

        return result * scale

    def __repr__(self) -> str:
        return (
            f"CACorrectRGB(radius={self._radius}, strength={self._strength}, "
            f"guide={self._guide})"
        )

    def __str__(self) -> str:
        guide_names = {GREEN: "green", RED: "red", BLUE: "blue"}
        return f"CA Correction using {guide_names.get(self._guide, 'unknown')} guide"

    def __len__(self) -> int:
        kernel = self._settings.kernel
        return kernel.size_multiplier * self._radius + kernel.size_offset

    def __bool__(self) -> bool:
        return self._strength > self._settings.numerical.clip_min

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CACorrectRGB):
            return NotImplemented
        return (
            self._radius == other._radius
            and self._strength == other._strength
            and self._guide == other._guide
        )

    def __hash__(self) -> int:
        return hash((self._radius, self._strength, self._guide))

    def __iter__(self):
        return iter(self._params().items())

    def __getitem__(self, key: str):
        return self._params().get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._params()

    def _params(self) -> dict:
        return {
            "radius": self._radius,
            "strength": self._strength,
            "guide": self._guide,
            "mode": self._mode,
            "refine": self._refine,
            "safety": self._safety,
        }

    def _validate_input(self, image: NDArray) -> NDArray:
        is_valid = image.ndim == 3 and image.shape[2] == CHANNEL_COUNT
        if not is_valid:
            raise ValueError("Input must be HxWx3 RGB image")
        return image.astype(np.float64)

    def _normalize(self, image: NDArray) -> tuple[NDArray, float]:
        max_val = image.max()
        threshold = self._settings.numerical.normalize_threshold
        if max_val > threshold:
            return image / max_val, max_val
        return image, threshold


class CorrectionPipeline:

    def __init__(self):
        self._stages: list[CACorrectRGB] = []
        self._heap: list[tuple] = []

    def __iadd__(self, stage: tuple[int, CACorrectRGB]) -> "CorrectionPipeline":
        priority, corrector = stage
        heapq.heappush(self._heap, (priority, len(self._stages), corrector))
        self._stages.append(corrector)
        return self

    def __call__(self, image: NDArray) -> NDArray:
        result = image.copy()
        heap_copy = self._heap.copy()
        while heap_copy:
            _, _, corrector = heapq.heappop(heap_copy)
            result = corrector(result)
        return result

    def __len__(self) -> int:
        return len(self._stages)

    def __iter__(self):
        return iter(self._stages)

    def __repr__(self) -> str:
        return f"CorrectionPipeline(stages={len(self._stages)})"