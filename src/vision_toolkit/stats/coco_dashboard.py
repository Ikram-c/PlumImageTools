from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def load_coco_annotations(json_path: str) -> dict[str, Any]:
    with open(json_path, "r") as f:
        return json.load(f)


def discover_annotation_properties(
    coco_data: dict[str, Any], sample_size: int = 100
) -> dict[str, str]:
    property_types: dict[str, str] = {}
    for ann in coco_data["annotations"][:sample_size]:
        for key, value in ann.items():
            if key in property_types:
                continue
            if isinstance(value, bool):
                property_types[key] = "boolean"
            elif isinstance(value, (int, float)):
                property_types[key] = "numerical"
            elif isinstance(value, str):
                property_types[key] = "categorical"
            elif isinstance(value, list):
                property_types[key] = "list"
            else:
                property_types[key] = "other"
    return property_types


def extract_annotation_data(coco_data: dict[str, Any]) -> pd.DataFrame:
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    skip_keys = {"id", "image_id", "category_id", "bbox", "area", "iscrowd",
                 "segmentation"}
    rows = []
    for ann in coco_data["annotations"]:
        row = {
            "annotation_id": ann.get("id"),
            "image_id": ann.get("image_id"),
            "category_id": ann.get("category_id"),
            "category_name": categories.get(ann.get("category_id"), "Unknown"),
            "area": ann.get("area"),
            "iscrowd": ann.get("iscrowd"),
        }
        if "bbox" in ann:
            x, y, w, h = ann["bbox"]
            row.update(
                {
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_width": w,
                    "bbox_height": h,
                    "bbox_area": w * h,
                    "bbox_aspect_ratio": w / h if h > 0 else 0,
                }
            )
        for key, value in ann.items():
            if key not in skip_keys:
                row.setdefault(key, value)
        rows.append(row)
    return pd.DataFrame(rows)


class DashboardBuilder:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.categories = sorted(df["category_name"].dropna().unique())
        self._traces: list[go.BaseTraceType] = []
        self._buttons: list[dict] = []
        self._view_builders: dict[
            str, Callable[[pd.DataFrame, str], Optional[go.BaseTraceType]]
        ] = {}
        self._register_default_views()

    def register_view(
        self,
        name: str,
        builder: Callable[[pd.DataFrame, str], Optional[go.BaseTraceType]],
    ) -> None:
        self._view_builders[name] = builder

    def _register_default_views(self) -> None:
        self.register_view("area_distribution", self._area_hist)
        self.register_view("aspect_ratio_distribution", self._ratio_hist)
        self.register_view("area_vs_aspect_ratio", self._area_ratio_scatter)
        self.register_view("annotations_per_image", self._per_image_hist)

    def _area_hist(
        self, cat_df: pd.DataFrame, category: str
    ) -> Optional[go.Histogram]:
        if "bbox_area" not in cat_df.columns:
            return None
        areas = cat_df["bbox_area"].dropna()
        if areas.empty:
            return None
        return go.Histogram(
            x=areas, nbinsx=30, name=f"{category} - Area Distribution"
        )

    def _ratio_hist(
        self, cat_df: pd.DataFrame, category: str
    ) -> Optional[go.Histogram]:
        if "bbox_aspect_ratio" not in cat_df.columns:
            return None
        ratios = cat_df["bbox_aspect_ratio"].dropna()
        if ratios.empty:
            return None
        return go.Histogram(
            x=ratios, nbinsx=30, name=f"{category} - Aspect Ratio Distribution"
        )

    def _area_ratio_scatter(
        self, cat_df: pd.DataFrame, category: str
    ) -> Optional[go.Scatter]:
        needed = {"bbox_area", "bbox_aspect_ratio"}
        if not needed.issubset(cat_df.columns):
            return None
        clean = cat_df.dropna(subset=list(needed))
        if clean.empty:
            return None
        return go.Scatter(
            x=clean["bbox_aspect_ratio"],
            y=clean["bbox_area"],
            mode="markers",
            marker={"opacity": 0.7},
            name=f"{category} - Area vs Aspect Ratio",
        )

    def _per_image_hist(
        self, cat_df: pd.DataFrame, category: str
    ) -> Optional[go.Histogram]:
        counts = cat_df.groupby("image_id").size()
        if counts.empty:
            return None
        return go.Histogram(
            x=counts.values,
            nbinsx=20,
            name=f"{category} - Annotations per Image",
        )

    def _add_trace(
        self, trace: go.BaseTraceType, label: str, layout: dict
    ) -> None:
        index = len(self._traces)
        trace.visible = index == 0
        self._traces.append(trace)
        visibility = [False] * index + [True]
        self._buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [{"visible": visibility}, layout],
            }
        )

    def _pad_visibility(self) -> None:
        total = len(self._traces)
        for button in self._buttons:
            visible = button["args"][0]["visible"]
            button["args"][0]["visible"] = visible + [False] * (
                total - len(visible)
            )

    def build(self) -> go.Figure:
        overview_counts = self.df["category_name"].value_counts()
        self._add_trace(
            go.Bar(
                x=overview_counts.index,
                y=overview_counts.values,
                name="All Categories",
            ),
            "All Categories Overview",
            {
                "title": "All Categories Overview",
                "xaxis": {"title": "Category"},
                "yaxis": {"title": "Count"},
            },
        )

        for category in self.categories:
            cat_df = self.df[self.df["category_name"] == category]
            for view_name, builder in self._view_builders.items():
                trace = builder(cat_df, category)
                if trace is None:
                    continue
                readable = view_name.replace("_", " ").title()
                self._add_trace(
                    trace,
                    f"{category} - {readable}",
                    {"title": f"{category} - {readable}"},
                )

        self._pad_visibility()
        fig = go.Figure(data=self._traces)
        fig.update_layout(
            title="COCO Annotations Dashboard",
            updatemenus=[
                {
                    "buttons": self._buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 0.1,
                    "xanchor": "left",
                    "y": 1.12,
                    "yanchor": "top",
                }
            ],
            height=600,
        )
        return fig


def analyze_categorical_property(
    df: pd.DataFrame, property_name: str, category_col: str = "category_name"
) -> pd.DataFrame:
    counts = (
        df.groupby([category_col, property_name]).size().reset_index(name="count")
    )
    category_totals = df.groupby(category_col).size()
    counts["percentage"] = counts.apply(
        lambda row: (row["count"] / category_totals[row[category_col]]) * 100,
        axis=1,
    )
    return counts


def analyze_numerical_property(
    df: pd.DataFrame, property_name: str, category_col: str = "category_name"
) -> Optional[pd.DataFrame]:
    clean_df = df[df[property_name].notna() & df[category_col].notna()]
    if clean_df.empty:
        return None
    stats = clean_df.groupby(category_col)[property_name].agg(
        ["count", "mean", "std", "min", "max", "median"]
    )
    quantiles = clean_df.groupby(category_col)[property_name].quantile(
        [0.25, 0.75]
    ).unstack()
    stats["q25"] = quantiles[0.25]
    stats["q75"] = quantiles[0.75]
    return stats.round(3)


def create_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in sorted(df["category_name"].unique()):
        cat_df = df[df["category_name"] == category]
        rows.append(
            {
                "category": category,
                "count": len(cat_df),
                "percentage": len(cat_df) / len(df) * 100,
                "avg_area": cat_df["bbox_area"].mean()
                if "bbox_area" in cat_df.columns
                else np.nan,
                "avg_aspect_ratio": cat_df["bbox_aspect_ratio"].mean()
                if "bbox_aspect_ratio" in cat_df.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_dashboard_export(json_path: str, output_dir: str) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    coco_data = load_coco_annotations(json_path)
    df = extract_annotation_data(coco_data)
    logger.info(
        "Loaded %d annotations across %d categories",
        len(df),
        df["category_name"].nunique(),
    )

    dashboard_path = out / "main_dashboard.html"
    DashboardBuilder(df).build().write_html(str(dashboard_path))

    summary_path = out / "category_summary.csv"
    create_summary_frame(df).to_csv(summary_path, index=False)

    return {
        "dashboard": str(dashboard_path),
        "summary": str(summary_path),
    }