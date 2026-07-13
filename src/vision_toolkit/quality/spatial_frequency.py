from __future__ import annotations

from typing import Any, Optional

import numpy as np

from vision_toolkit.quality.image_utils import to_grayscale, validate_image


class SpatialFrequencyAnalyzer:

    def __init__(self, n_direction_bins: int = 36):
        self.n_direction_bins = n_direction_bins
        self.results: dict[str, dict[str, Any]] = {}

    def analyze_frequency(self, image: np.ndarray, image_id: str) -> dict[str, Any]:
        validate_image(image)
        gray = to_grayscale(image)

        fft_shifted, magnitude, phase = self.compute_2d_fft(gray)
        freq_metrics = self._compute_frequency_metrics(gray, magnitude)
        directional = self._compute_directional_energy(magnitude, gray.shape)
        centroid = self._compute_frequency_centroid(
            magnitude, directional["distance_matrix"]
        )

        result = {
            "fft_result": fft_shifted,
            "magnitude_spectrum": magnitude,
            "log_magnitude_spectrum": np.log(magnitude + 1),
            "phase_spectrum": phase,
            "radial_profile": freq_metrics["radial_profile"],
            "low_freq_energy_ratio": freq_metrics["low_freq_energy_ratio"],
            "mid_freq_energy_ratio": freq_metrics["mid_freq_energy_ratio"],
            "high_freq_energy_ratio": freq_metrics["high_freq_energy_ratio"],
            "directional_energy": directional["directional_energy"],
            "dominant_direction": directional["dominant_direction"],
            "frequency_centroid": centroid,
        }
        self.results[image_id] = result
        return result

    def compute_2d_fft(
        self, image: np.ndarray, normalize: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if normalize:
            image = image.astype(np.float32) / 255.0
        fft = np.fft.fftshift(np.fft.fft2(image))
        return fft, np.abs(fft), np.angle(fft)

    def _distance_matrix(self, rows: int, cols: int) -> np.ndarray:
        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(v, u)
        return np.sqrt(U**2 + V**2)

    def _compute_frequency_metrics(
        self, image: np.ndarray, magnitude: np.ndarray
    ) -> dict[str, Any]:
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        D = self._distance_matrix(rows, cols)
        max_radius = int(np.sqrt(crow**2 + ccol**2))

        radii = D.astype(int).ravel()
        valid = radii < max_radius
        radial_profile = np.bincount(
            radii[valid], weights=magnitude.ravel()[valid], minlength=max_radius
        )
        radial_count = np.bincount(radii[valid], minlength=max_radius)
        radial_count[radial_count == 0] = 1
        radial_profile = radial_profile / radial_count

        total_energy = np.sum(magnitude**2)
        low_mask = D <= max_radius * 0.25
        high_mask = D >= max_radius * 0.75
        mid_mask = ~low_mask & ~high_mask

        return {
            "radial_profile": radial_profile,
            "low_freq_energy_ratio": float(
                np.sum(magnitude[low_mask] ** 2) / total_energy
            ),
            "mid_freq_energy_ratio": float(
                np.sum(magnitude[mid_mask] ** 2) / total_energy
            ),
            "high_freq_energy_ratio": float(
                np.sum(magnitude[high_mask] ** 2) / total_energy
            ),
        }

    def _compute_directional_energy(
        self, magnitude: np.ndarray, shape: tuple[int, int]
    ) -> dict[str, Any]:
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(v, u)
        angles = np.arctan2(V, U)

        bins = np.linspace(-np.pi, np.pi, self.n_direction_bins)
        directional_energy = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            mask = (angles >= bins[i]) & (angles < bins[i + 1])
            directional_energy[i] = np.sum(magnitude[mask] ** 2)

        return {
            "directional_energy": directional_energy,
            "dominant_direction": float(
                np.rad2deg(bins[np.argmax(directional_energy)])
            ),
            "distance_matrix": np.sqrt(U**2 + V**2),
        }

    def _compute_frequency_centroid(
        self, magnitude: np.ndarray, distance_matrix: np.ndarray
    ) -> float:
        total_mag = np.sum(magnitude)
        if total_mag == 0:
            return 0.0
        return float(np.sum(magnitude * distance_matrix) / total_mag)

    def get_results(self) -> dict[str, dict[str, Any]]:
        return self.results.copy()

    def get_analysis(self, image_id: str) -> Optional[dict[str, Any]]:
        return self.results.get(image_id)

    def clear_results(self) -> None:
        self.results.clear()