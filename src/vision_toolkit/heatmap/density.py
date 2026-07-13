from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import plotly.graph_objects as go
import yaml
from scipy.ndimage import gaussian_filter

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _hydrate(cls: type, mapping: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in mapping.items() if k in known})


@dataclass(frozen=True)
class CanvasSettings:
    width: float = 20.0
    height: float = 10.0


@dataclass(frozen=True)
class FigureSettings:
    width: int = 1000
    height: int = 800


@dataclass(frozen=True)
class VisualisationSettings:
    box_line_color: str = "rgba(255, 255, 255, 0.6)"
    box_line_width: int = 2
    box_opacity: float = 1.0
    marker_color: str = "white"
    marker_size: int = 8
    marker_opacity: float = 0.8
    marker_line_width: int = 1
    marker_line_color: str = "black"
    colorscale: str = "Viridis"
    template: str = "plotly_dark"


@dataclass(frozen=True)
class LayoutSettings:
    title: str = "Bounding Box Density Heatmap"
    x_axis_title: str = "X Coordinate"
    y_axis_title: str = "Y Coordinate"
    colorbar_title: str = "Density"


@dataclass(frozen=True)
class HistogramSettings:
    bin_divisor: float = 0.5
    default_sigma: float = 1.0


@dataclass(frozen=True)
class SmoothingSettings:
    low: float = 0.5
    medium: float = 1.0
    high: float = 2.0


@dataclass(frozen=True)
class MenuSettings:
    y_position: float = 1.15
    bg_color: str = "rgba(0,0,0,0.5)"
    border_color: str = "grey"
    font_color: str = "white"
    button_pad_r: int = 10
    button_pad_t: int = 10
    colorscale_x: float = 0.1
    visibility_x: float = 0.5
    smoothing_x: float = 0.9


@dataclass(frozen=True)
class HeatmapSettings:
    canvas: CanvasSettings = field(default_factory=CanvasSettings)
    figure: FigureSettings = field(default_factory=FigureSettings)
    visualisation: VisualisationSettings = field(
        default_factory=VisualisationSettings
    )
    layout: LayoutSettings = field(default_factory=LayoutSettings)
    histogram: HistogramSettings = field(default_factory=HistogramSettings)
    smoothing: SmoothingSettings = field(default_factory=SmoothingSettings)
    menu: MenuSettings = field(default_factory=MenuSettings)
    colorscales: tuple[str, ...] = ("Viridis", "Plasma", "Inferno", "Magma", "Hot")


def load_heatmap_settings(path: Path = CONFIG_PATH) -> HeatmapSettings:
    if not path.exists():
        return HeatmapSettings()
    raw = yaml.safe_load(path.read_text()) or {}
    return HeatmapSettings(
        canvas=_hydrate(CanvasSettings, raw.get("canvas", {})),
        figure=_hydrate(FigureSettings, raw.get("figure", {})),
        visualisation=_hydrate(VisualisationSettings, raw.get("visualisation", {})),
        layout=_hydrate(LayoutSettings, raw.get("layout", {})),
        histogram=_hydrate(HistogramSettings, raw.get("histogram", {})),
        smoothing=_hydrate(SmoothingSettings, raw.get("smoothing", {})),
        menu=_hydrate(MenuSettings, raw.get("menu", {})),
        colorscales=tuple(raw.get("colorscales", HeatmapSettings().colorscales)),
    )


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    def __iter__(self) -> Iterator[float]:
        yield self.x0
        yield self.y0
        yield self.x1
        yield self.y1


class CocoBoxLoader:

    def __init__(self, json_path: Path):
        self._json_path = json_path

    def load_boxes(self) -> list[tuple[float, float, float, float]]:
        with open(self._json_path, "r") as f:
            data = json.load(f)
        boxes = []
        for annotation in data.get("annotations", []):
            if "bbox" in annotation:
                x, y, w, h = annotation["bbox"]
                boxes.append((x, y, x + w, y + h))
        return boxes


class HeatmapData:

    def __init__(self, boxes: list[BoundingBox], settings: HeatmapSettings):
        self._boxes = boxes
        self._settings = settings
        self._centers = self._compute_centers()
        self._hist, self._xedges, self._yedges = self._compute_histogram()

    def _compute_centers(self) -> np.ndarray:
        if not self._boxes:
            return np.empty((0, 2))
        return np.array([[box.center_x, box.center_y] for box in self._boxes])

    def _compute_histogram(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        canvas = self._settings.canvas
        divisor = self._settings.histogram.bin_divisor
        bins_x = int(canvas.width / divisor)
        bins_y = int(canvas.height / divisor)
        x_values = self._centers[:, 0] if len(self._centers) > 0 else []
        y_values = self._centers[:, 1] if len(self._centers) > 0 else []
        return np.histogram2d(
            x_values,
            y_values,
            bins=[bins_x, bins_y],
            range=[[0, canvas.width], [0, canvas.height]],
        )

    def __len__(self) -> int:
        return len(self._boxes)

    @property
    def centers(self) -> np.ndarray:
        return self._centers

    @property
    def histogram(self) -> np.ndarray:
        return self._hist

    @property
    def x_centers(self) -> np.ndarray:
        return (self._xedges[:-1] + self._xedges[1:]) / 2

    @property
    def y_centers(self) -> np.ndarray:
        return (self._yedges[:-1] + self._yedges[1:]) / 2

    def smoothed_histogram(self, sigma: float) -> np.ndarray:
        return gaussian_filter(self._hist, sigma=sigma)


class BoxShapeFactory:

    def __init__(self, settings: HeatmapSettings):
        self._vis = settings.visualisation

    def __call__(self, box: BoundingBox, index: int) -> dict[str, Any]:
        return {
            "type": "rect",
            "x0": box.x0,
            "y0": box.y0,
            "x1": box.x1,
            "y1": box.y1,
            "line": {
                "color": self._vis.box_line_color,
                "width": self._vis.box_line_width,
            },
            "fillcolor": "rgba(255, 255, 255, 0)",
            "opacity": self._vis.box_opacity,
            "layer": "above",
            "name": f"Box {index + 1}",
        }


class MenuBuilder:

    def __init__(
        self,
        settings: HeatmapSettings,
        heatmap_data: HeatmapData,
        box_shapes: list[dict],
    ):
        self._settings = settings
        self._heatmap_data = heatmap_data
        self._box_shapes = box_shapes

    def _base_menu_style(self) -> dict:
        menu = self._settings.menu
        return {
            "pad": {"r": menu.button_pad_r, "t": menu.button_pad_t},
            "showactive": True,
            "yanchor": "top",
            "y": menu.y_position,
            "bgcolor": menu.bg_color,
            "bordercolor": menu.border_color,
            "font": {"color": menu.font_color},
        }

    def _colorscale_menu(self) -> dict:
        buttons = [
            {"args": [{"colorscale": [cs]}], "label": cs, "method": "restyle"}
            for cs in self._settings.colorscales
        ]
        menu = {
            "type": "dropdown",
            "direction": "down",
            "buttons": buttons,
            "x": self._settings.menu.colorscale_x,
            "xanchor": "left",
        }
        menu.update(self._base_menu_style())
        return menu

    def _visibility_menu(self) -> dict:
        buttons = [
            {
                "args": [{"visible": [True, True]}, {"shapes": self._box_shapes}],
                "label": "Show All",
                "method": "update",
            },
            {
                "args": [{"visible": [True, False]}, {"shapes": []}],
                "label": "Hide Points & Boxes",
                "method": "update",
            },
            {
                "args": [{"visible": [True, True]}, {"shapes": []}],
                "label": "Hide Boxes Only",
                "method": "update",
            },
            {
                "args": [{"visible": [True, False]}, {"shapes": self._box_shapes}],
                "label": "Hide Points Only",
                "method": "update",
            },
        ]
        menu = {
            "type": "buttons",
            "direction": "right",
            "buttons": buttons,
            "x": self._settings.menu.visibility_x,
            "xanchor": "center",
        }
        menu.update(self._base_menu_style())
        return menu

    def _smoothing_menu(self) -> dict:
        smoothing = self._settings.smoothing
        labelled_sigmas = [
            ("Low Smoothing", smoothing.low),
            ("Medium Smoothing", smoothing.medium),
            ("High Smoothing", smoothing.high),
        ]
        buttons = [
            {
                "args": [{"z": [self._heatmap_data.smoothed_histogram(sigma).T]}],
                "label": label,
                "method": "restyle",
            }
            for label, sigma in labelled_sigmas
        ]
        menu = {
            "type": "buttons",
            "direction": "right",
            "buttons": buttons,
            "x": self._settings.menu.smoothing_x,
            "xanchor": "right",
        }
        menu.update(self._base_menu_style())
        return menu

    def __iter__(self):
        yield self._colorscale_menu()
        yield self._visibility_menu()
        yield self._smoothing_menu()


class BoundingBoxHeatmapVisualiser:

    def __init__(self, settings: HeatmapSettings = None):
        self._settings = settings or load_heatmap_settings()
        self._boxes: list[BoundingBox] = []
        self._figure: go.Figure = go.Figure()

    def add_boxes(
        self, box_tuples: list[tuple[float, float, float, float]]
    ) -> None:
        self._boxes = [BoundingBox(*t) for t in box_tuples]

    def _create_empty_figure(self) -> None:
        canvas = self._settings.canvas
        self._figure.add_annotation(
            text="No Data Available",
            x=canvas.width / 2,
            y=canvas.height / 2,
            showarrow=False,
            font={"size": 24},
        )

    def _create_heatmap_trace(self, heatmap_data: HeatmapData) -> go.Heatmap:
        vis = self._settings.visualisation
        layout = self._settings.layout
        sigma = self._settings.histogram.default_sigma
        return go.Heatmap(
            z=heatmap_data.smoothed_histogram(sigma).T,
            x=heatmap_data.x_centers,
            y=heatmap_data.y_centers,
            colorscale=vis.colorscale,
            colorbar={"title": layout.colorbar_title},
            name="Heatmap",
            visible=True,
            hovertemplate=(
                "X: %{x:.1f}<br>Y: %{y:.1f}<br>"
                "Frequency: %{z:.2f}<extra></extra>"
            ),
        )

    def _create_scatter_trace(self, heatmap_data: HeatmapData) -> go.Scatter:
        vis = self._settings.visualisation
        return go.Scatter(
            x=heatmap_data.centers[:, 0],
            y=heatmap_data.centers[:, 1],
            mode="markers",
            marker={
                "color": vis.marker_color,
                "size": vis.marker_size,
                "opacity": vis.marker_opacity,
                "line": {
                    "width": vis.marker_line_width,
                    "color": vis.marker_line_color,
                },
            },
            name="Box Centers",
            visible=True,
            hovertemplate=(
                "Center X: %{x:.1f}<br>Center Y: %{y:.1f}<extra></extra>"
            ),
        )

    def _apply_layout(self, box_shapes: list[dict], menus: list[dict]) -> None:
        canvas = self._settings.canvas
        figure = self._settings.figure
        layout = self._settings.layout
        self._figure.update_layout(
            shapes=box_shapes,
            title=layout.title,
            xaxis={
                "title": layout.x_axis_title,
                "range": [0, canvas.width],
                "constrain": "domain",
            },
            yaxis={
                "title": layout.y_axis_title,
                "range": [0, canvas.height],
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            width=figure.width,
            height=figure.height,
            template=self._settings.visualisation.template,
            showlegend=True,
            updatemenus=menus,
        )

    def build(self) -> go.Figure:
        if len(self._boxes) == 0:
            self._create_empty_figure()
            return self._figure

        heatmap_data = HeatmapData(self._boxes, self._settings)
        shape_factory = BoxShapeFactory(self._settings)
        box_shapes = [shape_factory(box, i) for i, box in enumerate(self._boxes)]

        self._figure.add_trace(self._create_heatmap_trace(heatmap_data))
        self._figure.add_trace(self._create_scatter_trace(heatmap_data))

        menu_builder = MenuBuilder(self._settings, heatmap_data, box_shapes)
        self._apply_layout(box_shapes, list(menu_builder))
        return self._figure

    def show(self) -> None:
        self._figure.show()

    def save_html(self, path: Path) -> None:
        self._figure.write_html(str(path))

    def save_image(self, path: Path) -> None:
        figure = self._settings.figure
        self._figure.write_image(str(path), width=figure.width, height=figure.height)


class HeatmapApplication:

    def __init__(self, output_dir: Path, settings: HeatmapSettings = None):
        self._settings = settings or load_heatmap_settings()
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, coco_json_path: Path, show: bool = False) -> Path:
        boxes = CocoBoxLoader(coco_json_path).load_boxes()
        visualiser = BoundingBoxHeatmapVisualiser(self._settings)
        visualiser.add_boxes(boxes)
        visualiser.build()
        output_html = self._output_dir / "heatmap.html"
        visualiser.save_html(output_html)
        if show:
            visualiser.show()
        return output_html