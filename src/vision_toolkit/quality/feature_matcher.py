from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vision_toolkit.quality.image_utils import (
    to_grayscale,
    to_normalized_grayscale,
)


@dataclass(frozen=True)
class DetectedFeatures:
    xy: np.ndarray
    size: np.ndarray
    angle: np.ndarray
    response: np.ndarray
    octave: np.ndarray
    descriptors: Optional[np.ndarray]


DetectorFn = Callable[[np.ndarray], DetectedFeatures]
DETECTOR_REGISTRY: dict[str, DetectorFn] = {}


def register_detector(name: str):
    def decorator(fn: DetectorFn) -> DetectorFn:
        DETECTOR_REGISTRY[name] = fn
        return fn

    return decorator


def _empty_features() -> DetectedFeatures:
    return DetectedFeatures(
        xy=np.zeros((0, 2)),
        size=np.zeros(0),
        angle=np.zeros(0),
        response=np.zeros(0),
        octave=np.zeros(0, dtype=int),
        descriptors=None,
    )


@register_detector("sift")
def _detect_sift(gray: np.ndarray) -> DetectedFeatures:
    from skimage.feature import SIFT

    detector = SIFT()
    try:
        detector.detect_and_extract(gray)
    except RuntimeError:
        return _empty_features()

    keypoints = detector.keypoints
    xy = keypoints[:, ::-1].astype(np.float64)
    n = len(keypoints)
    return DetectedFeatures(
        xy=xy,
        size=np.asarray(detector.sigmas, dtype=np.float64),
        angle=np.degrees(np.asarray(detector.orientations, dtype=np.float64)),
        response=np.zeros(n),
        octave=np.asarray(detector.octaves, dtype=int),
        descriptors=detector.descriptors,
    )


@register_detector("orb")
def _detect_orb(gray: np.ndarray) -> DetectedFeatures:
    from skimage.feature import ORB

    detector = ORB()
    try:
        detector.detect_and_extract(gray)
    except RuntimeError:
        return _empty_features()

    keypoints = detector.keypoints
    xy = keypoints[:, ::-1].astype(np.float64)
    n = len(keypoints)
    return DetectedFeatures(
        xy=xy,
        size=np.asarray(detector.scales, dtype=np.float64),
        angle=np.degrees(np.asarray(detector.orientations, dtype=np.float64)),
        response=np.asarray(detector.responses, dtype=np.float64)
        if hasattr(detector, "responses")
        else np.zeros(n),
        octave=np.zeros(n, dtype=int),
        descriptors=detector.descriptors.astype(np.uint8)
        if detector.descriptors is not None
        else None,
    )


@register_detector("akaze_opencv")
def _detect_akaze(gray: np.ndarray) -> DetectedFeatures:
    import cv2

    gray_u8 = gray if gray.dtype == np.uint8 else (gray * 255).astype(np.uint8)
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(gray_u8, None)
    if not keypoints:
        return _empty_features()

    return DetectedFeatures(
        xy=np.array([kp.pt for kp in keypoints], dtype=np.float64),
        size=np.array([kp.size for kp in keypoints]),
        angle=np.array([kp.angle for kp in keypoints]),
        response=np.array([kp.response for kp in keypoints]),
        octave=np.array([kp.octave for kp in keypoints], dtype=int),
        descriptors=descriptors,
    )


class ImageFeatureMatcher:

    def __init__(
        self,
        img: np.ndarray,
        matcher: str = "sift",
        dataset_name: str = "unknown_dataset",
        image_name: str = "unknown_image",
    ):
        if matcher not in DETECTOR_REGISTRY:
            raise ValueError(
                f"Unknown matcher '{matcher}'. "
                f"Registered: {sorted(DETECTOR_REGISTRY)}"
            )
        self.img = img
        self.matcher = matcher
        self.dataset_name = dataset_name
        self.image_name = image_name

    def _prepared_gray(self, use_normalized: bool) -> np.ndarray:
        if use_normalized:
            gray = to_normalized_grayscale(self.img)
        else:
            gray = to_grayscale(self.img)
        return gray.astype(np.float64) / 255.0

    def detect(self, use_normalized: bool = True) -> DetectedFeatures:
        gray = self._prepared_gray(use_normalized)
        return DETECTOR_REGISTRY[self.matcher](gray)

    def keypoints_to_dataframe(self, use_normalized: bool = True) -> pd.DataFrame:
        features = self.detect(use_normalized)
        img_source = "Normalized" if use_normalized else "Original"

        df = pd.DataFrame(
            {
                "dataset_name": self.dataset_name,
                "image_name": self.image_name,
                "img_source": img_source,
                "detector": self.matcher,
                "keypoint_id": np.arange(len(features.xy)),
                "x_coord": features.xy[:, 0],
                "y_coord": features.xy[:, 1],
                "size": features.size,
                "angle": features.angle,
                "response": features.response,
                "octave": features.octave,
            }
        )
        if features.descriptors is not None:
            for i in range(features.descriptors.shape[1]):
                df[f"descriptor_{i}"] = features.descriptors[:, i]
        return df

    def save_keypoints_csv(
        self, output_dir: str = "tmp", use_normalized: bool = True
    ) -> str:
        df = self.keypoints_to_dataframe(use_normalized)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "normalized" if use_normalized else "original"
        output_path = (
            out_dir / f"{self.image_name}_{self.matcher}_{suffix}_keypoints.csv"
        )
        df.to_csv(output_path, index=False)
        return str(output_path)

    def plot_features(
        self, use_normalized: bool = True, output_path: str = None
    ) -> str:
        suffix = "normalized" if use_normalized else "original"
        if output_path is None:
            output_path = str(
                Path("tmp")
                / f"{self.image_name}_{self.matcher}_{suffix}_features.png"
            )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        features = self.detect(use_normalized)
        gray = self._prepared_gray(use_normalized)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(gray, cmap="gray")
        if len(features.xy) > 0:
            ax.scatter(
                features.xy[:, 0],
                features.xy[:, 1],
                s=np.maximum(features.size, 1.0) * 10,
                facecolors="none",
                edgecolors="lime",
                linewidths=0.8,
            )
        ax.set_title(
            f"{suffix.capitalize()} {self.image_name} with "
            f"{len(features.xy)} {self.matcher.upper()} features"
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def analyze_and_save_all_data(self, output_dir: str = "tmp") -> dict[str, Any]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "normalized_plot": self.plot_features(
                True,
                str(
                    out_dir
                    / f"{self.image_name}_{self.matcher}_normalized_features.png"
                ),
            ),
            "original_plot": self.plot_features(
                False,
                str(
                    out_dir
                    / f"{self.image_name}_{self.matcher}_original_features.png"
                ),
            ),
            "normalized_csv": self.save_keypoints_csv(output_dir, True),
            "original_csv": self.save_keypoints_csv(output_dir, False),
        }
        combined_df = pd.concat(
            [self.keypoints_to_dataframe(True), self.keypoints_to_dataframe(False)],
            ignore_index=True,
        )
        combined_path = (
            out_dir / f"{self.image_name}_{self.matcher}_all_keypoints.csv"
        )
        combined_df.to_csv(combined_path, index=False)
        results["combined_csv"] = str(combined_path)
        return results