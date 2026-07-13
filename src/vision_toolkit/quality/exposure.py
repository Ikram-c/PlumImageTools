from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from vision_toolkit.quality.image_utils import to_grayscale, validate_image


@dataclass(frozen=True)
class ExposureSettings:
    overexposure_threshold: float = 240.0
    underexposure_threshold: float = 15.0
    significant_percentage: float = 2.0
    channel_clip_high: int = 250
    channel_clip_low: int = 5
    shadow_bin: int = 85
    highlight_bin: int = 171
    dark_brightness: float = 85.0
    bright_brightness: float = 170.0
    good_range_utilization: float = 70.0


class ExposureAnalyzer:

    def __init__(self, settings: ExposureSettings = ExposureSettings()):
        self.settings = settings
        self.results: dict[str, dict[str, Any]] = {}

    def analyze_exposure(self, image: np.ndarray, image_id: str) -> dict[str, Any]:
        validate_image(image)
        analysis = {
            **self._calculate_luminance_metrics(image),
            **self._detect_exposure_issues(image),
            **self._calculate_histogram_metrics(image),
            **self._calculate_dynamic_range(image),
            **self._generate_exposure_recommendations(image),
        }
        self.results[image_id] = analysis
        return analysis

    def _calculate_luminance_metrics(self, image: np.ndarray) -> dict[str, float]:
        if image.ndim == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            luminance = (
                0.2126 * rgb[:, :, 0]
                + 0.7152 * rgb[:, :, 1]
                + 0.0722 * rgb[:, :, 2]
            )
            l_channel = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
            v_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
        else:
            luminance = l_channel = v_channel = image.astype(float)

        return {
            "avg_luminance": float(np.mean(luminance)),
            "median_luminance": float(np.median(luminance)),
            "std_luminance": float(np.std(luminance)),
            "avg_brightness_lab": float(np.mean(l_channel)),
            "avg_brightness_hsv": float(np.mean(v_channel)),
            "min_luminance": float(np.min(luminance)),
            "max_luminance": float(np.max(luminance)),
            "luminance_range": float(np.max(luminance) - np.min(luminance)),
        }

    def _detect_exposure_issues(self, image: np.ndarray) -> dict[str, Any]:
        gray = to_grayscale(image)
        total_pixels = gray.size

        overexposed = int(np.sum(gray >= self.settings.overexposure_threshold))
        underexposed = int(np.sum(gray <= self.settings.underexposure_threshold))
        over_pct = (overexposed / total_pixels) * 100
        under_pct = (underexposed / total_pixels) * 100
        significant = self.settings.significant_percentage

        channel_clipping = {}
        if image.ndim == 3:
            for name, data in zip(("blue", "green", "red"), cv2.split(image)):
                high = np.sum(data >= self.settings.channel_clip_high)
                low = np.sum(data <= self.settings.channel_clip_low)
                channel_clipping[f"{name}_clipped_high_percent"] = float(
                    (high / total_pixels) * 100
                )
                channel_clipping[f"{name}_clipped_low_percent"] = float(
                    (low / total_pixels) * 100
                )

        return {
            "overexposed_pixels": overexposed,
            "overexposure_percentage": float(over_pct),
            "is_overexposed": over_pct > significant,
            "underexposed_pixels": underexposed,
            "underexposure_percentage": float(under_pct),
            "is_underexposed": under_pct > significant,
            "has_exposure_issues": over_pct > significant or under_pct > significant,
            **channel_clipping,
        }

    def _calculate_histogram_metrics(self, image: np.ndarray) -> dict[str, Any]:
        gray = to_grayscale(image)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / np.sum(hist)

        shadows = np.sum(hist_norm[: self.settings.shadow_bin])
        midtones = np.sum(
            hist_norm[self.settings.shadow_bin : self.settings.highlight_bin]
        )
        highlights = np.sum(hist_norm[self.settings.highlight_bin :])

        bins = np.arange(256)
        weighted_avg = np.sum(bins * hist_norm)
        nonzero = hist_norm[hist_norm > 0]
        entropy = -np.sum(nonzero * np.log2(nonzero))

        if shadows > 0.5:
            distribution = "shadows-heavy"
        elif highlights > 0.5:
            distribution = "highlights-heavy"
        else:
            distribution = "balanced"

        return {
            "shadows_percentage": float(shadows * 100),
            "midtones_percentage": float(midtones * 100),
            "highlights_percentage": float(highlights * 100),
            "histogram_center": float(weighted_avg),
            "histogram_entropy": float(entropy),
            "histogram_peak": int(np.argmax(hist)),
            "tonal_distribution": distribution,
        }

    def _calculate_dynamic_range(self, image: np.ndarray) -> dict[str, Any]:
        gray = to_grayscale(image)
        p1, p5, p95, p99 = np.percentile(gray, (1, 5, 95, 99))
        full_range = float(np.max(gray) - np.min(gray))
        effective_99 = float(p99 - p1)
        utilization = (effective_99 / 255.0) * 100

        return {
            "dynamic_range_full": full_range,
            "dynamic_range_99p": effective_99,
            "dynamic_range_95p": float(p95 - p5),
            "range_utilization_percent": float(utilization),
            "p1_percentile": float(p1),
            "p99_percentile": float(p99),
            "has_good_dynamic_range": utilization
            > self.settings.good_range_utilization,
        }

    def _generate_exposure_recommendations(self, image: np.ndarray) -> dict[str, Any]:
        gray = to_grayscale(image)
        avg_brightness = float(np.mean(gray))
        over_pct = (
            np.sum(gray >= self.settings.overexposure_threshold) / gray.size
        ) * 100
        under_pct = (
            np.sum(gray <= self.settings.underexposure_threshold) / gray.size
        ) * 100

        exposure_comp, recs = self._compensation_for(
            over_pct, under_pct, avg_brightness
        )

        if len(recs) == 1 and "well-balanced" in recs[0]:
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            if np.sum(hist[:50]) > np.sum(hist[200:]):
                recs.append(
                    "Consider slight exposure increase for better shadow detail"
                )
            elif np.sum(hist[200:]) > np.sum(hist[:50]):
                recs.append(
                    "Consider slight exposure decrease to prevent highlight clipping"
                )

        return {
            "exposure_compensation_stops": float(exposure_comp),
            "recommendations": recs,
            "overall_exposure_quality": self._assess_exposure_quality(
                over_pct, under_pct, avg_brightness
            ),
        }

    def _compensation_for(
        self, over_pct: float, under_pct: float, avg_brightness: float
    ) -> tuple[float, list[str]]:
        significant = self.settings.significant_percentage
        tiers = [
            (10, 2.0, "Significantly"),
            (5, 1.0, ""),
            (significant, 0.5, "Slightly"),
        ]
        if over_pct > significant:
            for limit, stops, label in tiers:
                if over_pct > limit:
                    prefix = f"{label} overexposed".strip().capitalize()
                    return -stops, [f"{prefix} - reduce exposure by {stops} stops"]
        if under_pct > significant:
            for limit, stops, label in tiers:
                if under_pct > limit:
                    prefix = f"{label} underexposed".strip().capitalize()
                    return stops, [f"{prefix} - increase exposure by {stops} stops"]
        if avg_brightness < self.settings.dark_brightness:
            return 0.5, ["Image appears dark - consider increasing exposure"]
        if avg_brightness > self.settings.bright_brightness:
            return -0.5, ["Image appears bright - consider reducing exposure"]
        return 0.0, ["Exposure appears well-balanced"]

    def _assess_exposure_quality(
        self, over_pct: float, under_pct: float, avg_brightness: float
    ) -> str:
        if over_pct > 10 or under_pct > 10:
            return "Poor"
        if over_pct > 5 or under_pct > 5:
            return "Fair"
        is_excellent = (
            over_pct < 1
            and under_pct < 1
            and self.settings.dark_brightness
            <= avg_brightness
            <= self.settings.bright_brightness
        )
        if is_excellent:
            return "Excellent"
        return "Good"

    def get_results(self) -> dict[str, dict[str, Any]]:
        return self.results.copy()

    def get_analysis(self, image_id: str) -> Optional[dict[str, Any]]:
        return self.results.get(image_id)

    def clear_results(self) -> None:
        self.results.clear()


def get_brightness_summary(image: np.ndarray) -> dict[str, Any]:
    analyzer = ExposureAnalyzer()
    luminance = analyzer._calculate_luminance_metrics(image)
    histogram = analyzer._calculate_histogram_metrics(image)
    dynamic_range = analyzer._calculate_dynamic_range(image)
    return {
        "average_brightness": luminance["avg_luminance"],
        "brightness_distribution": histogram["tonal_distribution"],
        "dynamic_range_utilization": dynamic_range["range_utilization_percent"],
        "histogram_center": histogram["histogram_center"],
    }