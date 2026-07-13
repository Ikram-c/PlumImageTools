from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.feature import match_descriptors

from vision_toolkit.quality.feature_matcher import (
    DetectedFeatures,
    ImageFeatureMatcher,
)

logger = logging.getLogger(__name__)


class HomographyEstimator:

    def __init__(
        self,
        detector: str = "sift",
        use_normalized: bool = True,
        min_match_count: int = 10,
        ransac_reproj_threshold: float = 5.0,
        ransac_confidence: float = 0.99,
        ransac_max_iters: int = 2000,
    ):
        self.detector = detector
        self.use_normalized = use_normalized
        self.min_match_count = min_match_count
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.ransac_confidence = ransac_confidence
        self.ransac_max_iters = ransac_max_iters

    def estimate(self, orig_img: np.ndarray, corrected_img: np.ndarray) -> np.ndarray:
        feats1 = self._detect(orig_img, "original")
        feats2 = self._detect(corrected_img, "corrected")
        self._validate_features(feats1, feats2)

        matches = match_descriptors(
            feats1.descriptors, feats2.descriptors, cross_check=True
        )
        if len(matches) < 4:
            raise ValueError(f"Not enough matches found: {len(matches)} < 4")
        if len(matches) < self.min_match_count:
            logger.warning(
                "Only %d matches found (wanted %d); proceeding.",
                len(matches),
                self.min_match_count,
            )

        src_pts = feats1.xy[matches[:, 0]].astype(np.float32).reshape(-1, 1, 2)
        dst_pts = feats2.xy[matches[:, 1]].astype(np.float32).reshape(-1, 1, 2)

        matrix, _ = cv2.findHomography(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
            confidence=self.ransac_confidence,
            maxIters=self.ransac_max_iters,
        )
        if matrix is None:
            raise ValueError("Could not compute transformation matrix")
        return matrix

    def _detect(self, img: np.ndarray, label: str) -> DetectedFeatures:
        matcher = ImageFeatureMatcher(img, matcher=self.detector, image_name=label)
        return matcher.detect(use_normalized=self.use_normalized)

    def _validate_features(
        self, feats1: DetectedFeatures, feats2: DetectedFeatures
    ) -> None:
        no_descriptors = feats1.descriptors is None or feats2.descriptors is None
        no_keypoints = len(feats1.xy) == 0 or len(feats2.xy) == 0
        if no_descriptors or no_keypoints:
            raise ValueError("Could not extract features from one or both images")


class AnnotationTransformer:

    def __init__(
        self,
        coco_json_path: str,
        orig_img_dir: str,
        corrected_img_dir: str,
        estimator: HomographyEstimator = None,
    ):
        self.coco_json_path = Path(coco_json_path)
        self.orig_img_dir = Path(orig_img_dir)
        self.corrected_img_dir = Path(corrected_img_dir)
        self.estimator = estimator or HomographyEstimator()

        with open(self.coco_json_path, "r") as f:
            self.coco_data = json.load(f)

        self.id_to_image = {
            img["id"]: img for img in self.coco_data.get("images", [])
        }
        self.orig_to_corrected = self._build_orig_to_corrected_map()
        self.transformation_cache: dict[tuple[str, str], np.ndarray] = {}

    def _build_orig_to_corrected_map(self) -> dict[str, Path]:
        corrected_files = {
            p.name: p for p in self.corrected_img_dir.rglob("*") if p.is_file()
        }
        mapping = {}
        for img in self.coco_data.get("images", []):
            base_name = Path(img["file_name"]).name
            if base_name in corrected_files:
                mapping[base_name] = corrected_files[base_name]
            else:
                logger.warning("Corrected version for %s not found.", base_name)
        return mapping

    def compute_transformation_matrix(
        self, orig_path: Path, corrected_path: Path
    ) -> np.ndarray:
        cache_key = (str(orig_path), str(corrected_path))
        if cache_key in self.transformation_cache:
            return self.transformation_cache[cache_key]

        orig_img = cv2.imread(str(orig_path))
        corrected_img = cv2.imread(str(corrected_path))
        if orig_img is None or corrected_img is None:
            raise FileNotFoundError(
                f"Could not load one of: {orig_path}, {corrected_path}"
            )

        matrix = self.estimator.estimate(orig_img, corrected_img)
        self.transformation_cache[cache_key] = matrix
        return matrix

    def transform_point(
        self, matrix: np.ndarray, point: tuple[float, float]
    ) -> tuple[float, float]:
        src = np.array([[point[0], point[1]]], dtype=np.float32).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, matrix)
        return float(dst[0][0][0]), float(dst[0][0][1])

    def clamp_point(
        self, point: tuple[float, float], width: int, height: int
    ) -> tuple[float, float]:
        x, y = point
        return max(0, min(x, width - 1)), max(0, min(y, height - 1))

    def transform_bbox(
        self, matrix: np.ndarray, bbox: list[float], width: int, height: int
    ) -> Optional[list[float]]:
        x, y, w, h = bbox
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        transformed = [
            self.clamp_point(self.transform_point(matrix, c), width, height)
            for c in corners
        ]
        x_coords = [pt[0] for pt in transformed]
        y_coords = [pt[1] for pt in transformed]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        if bbox_w <= 1 or bbox_h <= 1:
            return None
        return [min_x, min_y, bbox_w, bbox_h]

    def transform_segmentation(
        self,
        matrix: np.ndarray,
        segmentation: list[list[float]],
        width: int,
        height: int,
    ) -> list[list[float]]:
        transformed_segmentation = []
        for polygon in segmentation:
            points = [
                (polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)
            ]
            transformed = [
                self.clamp_point(self.transform_point(matrix, pt), width, height)
                for pt in points
            ]
            flat = [coord for pt in transformed for coord in pt]
            if len(flat) >= 6 and self._polygon_area(transformed) > 1:
                transformed_segmentation.append(flat)
        return transformed_segmentation

    def _polygon_area(self, vertices: list[tuple[float, float]]) -> float:
        if len(vertices) < 3:
            return 0.0
        x = np.array([v[0] for v in vertices])
        y = np.array([v[1] for v in vertices])
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def transform_keypoints(
        self, matrix: np.ndarray, keypoints: list[float], width: int, height: int
    ) -> list[float]:
        if len(keypoints) % 3 != 0:
            raise ValueError("Keypoints must be in groups of 3 (x,y,visibility)")
        transformed = []
        for i in range(0, len(keypoints), 3):
            x, y, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]
            if visibility > 0:
                tx, ty = self.transform_point(matrix, (x, y))
                tx, ty = self.clamp_point((tx, ty), width, height)
                transformed.extend([tx, ty, visibility])
            else:
                transformed.extend([x, y, visibility])
        return transformed

    def transform_dataset_annotations(self, output_json_path: str) -> dict:
        coco_out = json.loads(json.dumps(self.coco_data))

        new_images = []
        imgid_to_size = {}
        for img in coco_out["images"]:
            base_name = Path(img["file_name"]).name
            if base_name not in self.orig_to_corrected:
                logger.warning(
                    "Skipping image %s, no corrected version found.", base_name
                )
                continue
            corrected_path = self.orig_to_corrected[base_name]
            corrected_img = cv2.imread(str(corrected_path))
            if corrected_img is None:
                logger.warning(
                    "Could not load corrected image for %s. Keeping original size.",
                    base_name,
                )
                height, width = img["height"], img["width"]
            else:
                height, width = corrected_img.shape[:2]
            new_entry = dict(img)
            new_entry["file_name"] = corrected_path.name
            new_entry["width"] = width
            new_entry["height"] = height
            new_images.append(new_entry)
            imgid_to_size[img["id"]] = (width, height)
        coco_out["images"] = new_images

        new_annotations = []
        skipped = 0
        for ann in coco_out["annotations"]:
            result = self._transform_single_annotation(ann, imgid_to_size)
            if result is not None:
                new_annotations.append(result)
            else:
                skipped += 1
        coco_out["annotations"] = new_annotations

        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(coco_out, f, indent=2)

        stats = {
            "images_transformed": len(new_images),
            "annotations_kept": len(new_annotations),
            "annotations_skipped": skipped,
            "homographies_computed": len(self.transformation_cache),
            "output_path": str(output_path),
        }
        logger.info("Transformed COCO JSON saved to %s", output_path)
        return stats

    def _transform_single_annotation(
        self, ann: dict, imgid_to_size: dict[int, tuple[int, int]]
    ) -> Optional[dict]:
        img_id = ann["image_id"]
        if img_id not in self.id_to_image or img_id not in imgid_to_size:
            logger.warning(
                "Annotation %s refers to missing image id %s. Skipping.",
                ann["id"],
                img_id,
            )
            return None

        base_name = Path(self.id_to_image[img_id]["file_name"]).name
        if base_name not in self.orig_to_corrected:
            return None

        orig_path = self.orig_img_dir / base_name
        corrected_path = self.orig_to_corrected[base_name]
        width, height = imgid_to_size[img_id]

        try:
            matrix = self.compute_transformation_matrix(orig_path, corrected_path)
        except Exception as e:
            logger.warning(
                "Failed to compute transformation for %s: %s. Skipping annotation %s.",
                base_name,
                e,
                ann["id"],
            )
            return None

        if "bbox" in ann:
            new_bbox = self.transform_bbox(matrix, ann["bbox"], width, height)
            if new_bbox is None:
                return None
            ann["bbox"] = new_bbox

        if "segmentation" in ann and isinstance(ann["segmentation"], list):
            new_seg = self.transform_segmentation(
                matrix, ann["segmentation"], width, height
            )
            if not new_seg:
                return None
            ann["segmentation"] = new_seg

        if "keypoints" in ann:
            ann["keypoints"] = self.transform_keypoints(
                matrix, ann["keypoints"], width, height
            )
        return ann