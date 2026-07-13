from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

_META_COLUMNS = ("category_id", "annotation_id", "image_id")


def _empty_segmentation_features() -> dict[str, float]:
    return {
        "segmentation_points": 0,
        "segmentation_area": 0,
        "bbox_to_seg_ratio": 1,
        "segmentation_perimeter": 0,
        "seg_circularity": 0,
    }


class AnnotationFeatureExtractor:

    def __init__(self):
        self.feature_names: list[str] = []

    def extract_bbox_features(self, annotations: list[dict]) -> pd.DataFrame:
        features = []
        for ann in tqdm(annotations, desc="Extracting bbox features"):
            if "bbox" not in ann:
                continue
            x, y, w, h = ann["bbox"]
            feature_dict = {
                "bbox_width": w,
                "bbox_height": h,
                "bbox_area": w * h,
                "bbox_aspect_ratio": w / h if h > 0 else 0,
                "bbox_x_center": x + w / 2,
                "bbox_y_center": y + h / 2,
                "bbox_perimeter": 2 * (w + h),
                "bbox_diagonal": np.sqrt(w**2 + h**2),
                "bbox_compactness": (4 * np.pi * w * h) / ((2 * (w + h)) ** 2)
                if w + h > 0
                else 0,
                "bbox_elongation": max(w, h) / min(w, h) if min(w, h) > 0 else 1,
                "bbox_rectangularity": (w * h) / ((w + h) / 2) ** 2
                if w + h > 0
                else 0,
                "category_id": ann.get("category_id", 0),
                "annotation_id": ann.get("id", 0),
                "image_id": ann.get("image_id", 0),
            }
            feature_dict.update(
                self._segmentation_features(ann, feature_dict["bbox_area"])
            )
            features.append(feature_dict)

        df = pd.DataFrame(features)
        self.feature_names = [c for c in df.columns if c not in _META_COLUMNS]
        return df

    def _segmentation_features(
        self, ann: dict, bbox_area: float
    ) -> dict[str, float]:
        seg = ann.get("segmentation")
        if not seg or not isinstance(seg, list) or not seg:
            return _empty_segmentation_features()
        try:
            polygon = seg[0]
            if len(polygon) < 6:
                return _empty_segmentation_features()
            seg_array = np.array(polygon).reshape(-1, 2)
            area = self._polygon_area(seg_array)
            perimeter = self._polygon_perimeter(seg_array)
            return {
                "segmentation_points": len(seg_array),
                "segmentation_area": area,
                "bbox_to_seg_ratio": bbox_area / area if area > 0 else 1,
                "segmentation_perimeter": perimeter,
                "seg_circularity": (4 * np.pi * area) / (perimeter**2)
                if perimeter > 0
                else 0,
            }
        except (ValueError, IndexError, TypeError):
            return _empty_segmentation_features()

    def _polygon_area(self, vertices: np.ndarray) -> float:
        if len(vertices) < 3:
            return 0.0
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * abs(
            sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1))
        )

    def _polygon_perimeter(self, vertices: np.ndarray) -> float:
        if len(vertices) < 2:
            return 0.0
        perimeter = 0.0
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            perimeter += np.sqrt(
                (vertices[j][0] - vertices[i][0]) ** 2
                + (vertices[j][1] - vertices[i][1]) ** 2
            )
        return perimeter

    def extract_spatial_features(
        self, annotations: list[dict], image_info: dict[int, dict]
    ) -> pd.DataFrame:
        features = []
        for ann in tqdm(annotations, desc="Extracting spatial features"):
            if "bbox" not in ann or ann["image_id"] not in image_info:
                continue
            x, y, w, h = ann["bbox"]
            img_info = image_info[ann["image_id"]]
            img_width = img_info.get("width", 1)
            img_height = img_info.get("height", 1)
            diag = np.sqrt(img_width**2 + img_height**2)

            features.append(
                {
                    "relative_x": x / img_width,
                    "relative_y": y / img_height,
                    "relative_width": w / img_width,
                    "relative_height": h / img_height,
                    "relative_area": (w * h) / (img_width * img_height),
                    "distance_from_center": np.sqrt(
                        ((x + w / 2) - img_width / 2) ** 2
                        + ((y + h / 2) - img_height / 2) ** 2
                    )
                    / diag,
                    "edge_proximity": min(
                        x, y, img_width - (x + w), img_height - (y + h)
                    )
                    / min(img_width, img_height),
                    "corner_distance_tl": np.sqrt(x**2 + y**2) / diag,
                    "corner_distance_br": np.sqrt(
                        (img_width - (x + w)) ** 2 + (img_height - (y + h)) ** 2
                    )
                    / diag,
                    "horizontal_position": (x + w / 2) / img_width,
                    "vertical_position": (y + h / 2) / img_height,
                    "category_id": ann.get("category_id", 0),
                    "annotation_id": ann.get("id", 0),
                    "image_id": ann.get("image_id", 0),
                }
            )

        df = pd.DataFrame(features)
        spatial = [c for c in df.columns if c not in _META_COLUMNS]
        self.feature_names.extend(spatial)
        return df


def load_coco_annotations(json_path: str) -> tuple[list[dict], dict[int, dict]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    annotations = data.get("annotations", [])
    images = data.get("images", [])
    image_info = {img["id"]: img for img in images}
    return annotations, image_info


def extract_combined_features(json_path: str) -> pd.DataFrame:
    annotations, image_info = load_coco_annotations(json_path)
    extractor = AnnotationFeatureExtractor()
    bbox_features = extractor.extract_bbox_features(annotations)
    spatial_features = extractor.extract_spatial_features(annotations, image_info)
    return bbox_features.merge(
        spatial_features, on=list(_META_COLUMNS), how="inner"
    )