from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def recommend_anchor_boxes(
    cluster_centers: np.ndarray, feature_names: list[str], n: int = 6
) -> np.ndarray:
    width_idx = (
        feature_names.index("bbox_width")
        if "bbox_width" in feature_names
        else None
    )
    height_idx = (
        feature_names.index("bbox_height")
        if "bbox_height" in feature_names
        else None
    )
    if width_idx is None or height_idx is None:
        raise ValueError(
            "bbox_width and bbox_height features required for anchor recommendation."
        )

    anchors = cluster_centers[:, [width_idx, height_idx]]
    areas = anchors[:, 0] * anchors[:, 1]
    anchors_sorted = anchors[np.argsort(-areas)]

    unique_anchors: list[np.ndarray] = []
    for box in anchors_sorted:
        is_duplicate = any(
            np.allclose(box, ua, rtol=0.1) for ua in unique_anchors
        )
        if not is_duplicate:
            unique_anchors.append(box)
        if len(unique_anchors) >= n:
            break
    return np.array(unique_anchors)


def suggest_augmentations(feature_importance: dict[str, float]) -> list[str]:
    recs = []
    spatial_feats = (
        "bbox_x_center",
        "bbox_y_center",
        "horizontal_position",
        "vertical_position",
    )
    area_feats = ("bbox_area", "bbox_width", "bbox_height", "bbox_aspect_ratio")

    has_spatial = any(feature_importance.get(f, 0) > 0.2 for f in spatial_feats)
    has_area = any(feature_importance.get(f, 0) > 0.2 for f in area_feats)
    has_elongation = feature_importance.get("bbox_elongation", 0) > 0.2

    if has_spatial:
        recs.append("Apply random translations, crops, and spatial flips.")
    if has_area:
        recs.append("Apply random scaling and aspect ratio jitter.")
    if has_elongation:
        recs.append("Consider random rotations and stretching.")
    if not recs:
        recs.append("Standard color and geometric augmentations recommended.")
    return recs


def recommend_class_weights(
    cluster_labels: np.ndarray,
    df: pd.DataFrame,
    feature_importance: Optional[dict[str, float]] = None,
) -> list[str]:
    if feature_importance is not None and feature_importance:
        mean_importance = np.mean(list(feature_importance.values()))
        hard_features = [
            k for k, v in feature_importance.items() if v < mean_importance / 2
        ]
    else:
        hard_features = []

    df = df.copy()
    df["cluster"] = cluster_labels
    hard_clusters = []
    for c in np.unique(cluster_labels):
        cdf = df[df["cluster"] == c]
        if cdf.shape[0] < 10:
            continue
        cat_counts = cdf["category_id"].value_counts(normalize=True)
        if cat_counts.max() < 0.7:
            hard_clusters.append(c)

    advice = []
    if hard_clusters:
        advice.append(
            f"Increase loss weight for objects in clusters: {hard_clusters} "
            "(these are hard to separate)."
        )
        if not df["category_id"].isnull().all():
            for c in hard_clusters:
                cats = df[df["cluster"] == c]["category_id"].unique()
                advice.append(
                    f"Cluster {c} mixes categories {cats}. "
                    "Consider more data or cleaning for these."
                )
    if hard_features:
        advice.append(
            f"Objects differing mostly by {hard_features} are difficult to "
            "separate; consider more varied examples or domain-specific "
            "augmentations."
        )
    if not advice:
        advice.append(
            "No particularly hard-to-separate clusters detected. "
            "Use standard class balancing."
        )
    return advice


def summarise_optimizer_recommendations(
    result: Any, feature_df: pd.DataFrame
) -> dict[str, Any]:
    return {
        "best_algorithm": result.algorithm,
        "n_clusters": result.n_clusters,
        "top_features": sorted(
            result.feature_importance.items(), key=lambda x: -x[1]
        )[:8],
        "anchor_boxes": recommend_anchor_boxes(
            result.cluster_centers, result.feature_names
        ).tolist()
        if result.cluster_centers is not None
        else [],
        "augmentations": suggest_augmentations(result.feature_importance),
        "class_weight_advice": recommend_class_weights(
            result.cluster_labels,
            feature_df,
            feature_importance=result.feature_importance,
        ),
    }