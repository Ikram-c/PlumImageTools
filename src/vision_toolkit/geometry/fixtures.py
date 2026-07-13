from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PolygonFixture:
    name: str
    points: np.ndarray
    bbox: tuple[float, float, float, float]
    is_self_intersecting: bool


def self_intersecting_eight() -> PolygonFixture:
    return PolygonFixture(
        name="self_intersecting_eight",
        points=np.array(
            [[0.0, 0.0], [3.0, 0.0], [2.0, 1.0], [0.0, 3.0], [3.0, 2.0]]
        ),
        bbox=(0.0, 0.0, 3.0, 2.0),
        is_self_intersecting=True,
    )


def separated_shapes(
    seed: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = [(2.0, 3.0), (6.0, 3.0), (10.0, 3.0)]

    rect = _rotated_rectangle(
        centers[0], rng.uniform(1, 2), rng.uniform(1, 2), rng.uniform(0, 360)
    )
    triangle = _rotated_right_triangle(
        centers[1],
        rng.uniform(1.5, 2.5),
        rng.uniform(1.5, 2.5),
        rng.uniform(0, 360),
    )
    hexagon = _regular_hexagon(
        centers[2], rng.uniform(0.8, 1.5), rng.uniform(0, 360)
    )
    return rect, triangle, hexagon


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _rotated_rectangle(
    center: tuple[float, float], width: float, height: float, angle_deg: float
) -> np.ndarray:
    v_rel = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )
    return v_rel @ _rotation_matrix(angle_deg).T + np.asarray(center)


def _rotated_right_triangle(
    center: tuple[float, float], base: float, height: float, angle_deg: float
) -> np.ndarray:
    v = np.array([[0.0, 0.0], [base, 0.0], [0.0, height]])
    v_centered = v - v.mean(axis=0)
    return v_centered @ _rotation_matrix(angle_deg).T + np.asarray(center)


def _regular_hexagon(
    center: tuple[float, float], radius: float, orientation_deg: float
) -> np.ndarray:
    angles = np.radians(orientation_deg) + np.linspace(
        0, 2 * np.pi, 6, endpoint=False
    )
    return np.stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        ],
        axis=1,
    )


def plot_separated_shapes(seed: int = 101) -> None:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    rect, triangle, hexagon = separated_shapes(seed)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    rng = np.random.default_rng(seed)
    shapes = ((rect, "Rectangle"), (triangle, "Triangle"), (hexagon, "Hexagon"))
    for verts, label in shapes:
        ax.add_patch(
            patches.Polygon(
                verts,
                closed=True,
                linewidth=2,
                edgecolor="black",
                facecolor=rng.random(3),
                alpha=0.6,
                label=label,
            )
        )

    ax.legend(loc="upper right")
    ax.set_title(f"Non-Intersecting Shapes (Seed: {seed})")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def generate_nested_rectangles(
    n_groups: int = 5,
    max_depth: int = 3,
    bounds: tuple = (0, 100, 0, 100),
    min_size: float = 5.0,
    margin_range: tuple = (2.0, 10.0),
    seed: Optional[int] = None,
) -> tuple:
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = bounds

    rectangles = []
    nesting_groups = []
    expected_nested = []

    for _ in range(n_groups):
        depth = rng.integers(1, max_depth + 1)
        group_indices = []

        width = rng.uniform(min_size * depth * 2, (x_max - x_min) / 2)
        height = rng.uniform(min_size * depth * 2, (y_max - y_min) / 2)
        x0 = rng.uniform(x_min, x_max - width)
        y0 = rng.uniform(y_min, y_max - height)

        current_rect = (x0, y0, x0 + width, y0 + height)
        rect_idx = len(rectangles)
        rectangles.append(current_rect)
        group_indices.append(rect_idx)

        for _ in range(1, depth):
            margin = rng.uniform(*margin_range)
            x0_new = current_rect[0] + margin
            y0_new = current_rect[1] + margin
            x1_new = current_rect[2] - margin
            y1_new = current_rect[3] - margin

            too_small = (
                x1_new - x0_new < min_size or y1_new - y0_new < min_size
            )
            if too_small:
                break

            current_rect = (x0_new, y0_new, x1_new, y1_new)
            rect_idx = len(rectangles)
            rectangles.append(current_rect)
            group_indices.append(rect_idx)
            expected_nested.append(rect_idx)

        nesting_groups.append(group_indices)

    return rectangles, sorted(expected_nested), nesting_groups


def generate_mixed_test_rectangles(
    n_nested_groups: int = 3,
    n_overlapping: int = 4,
    n_isolated: int = 3,
    bounds: tuple = (0, 100, 0, 100),
    seed: Optional[int] = None,
) -> dict:
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = bounds

    nested_rects, expected_nested, nesting_groups = generate_nested_rectangles(
        n_groups=n_nested_groups,
        max_depth=3,
        bounds=(x_min, x_max * 0.4, y_min, y_max),
        seed=seed,
    )
    rectangles = list(nested_rects)

    overlap_start_idx = len(rectangles)
    overlap_center_x = x_max * 0.7
    overlap_center_y = y_max * 0.5
    for _ in range(n_overlapping):
        size = rng.uniform(10, 25)
        x0 = overlap_center_x + rng.uniform(-15, 15)
        y0 = overlap_center_y + rng.uniform(-15, 15)
        rectangles.append((x0, y0, x0 + size, y0 + size))
    overlapping_indices = list(range(overlap_start_idx, len(rectangles)))

    isolated_start_idx = len(rectangles)
    for i in range(n_isolated):
        size_x = rng.uniform(5, 15)
        size_y = rng.uniform(5, 15)
        x0 = rng.uniform(x_min + i * 20, x_min + i * 20 + 15)
        y0 = rng.uniform(y_max * 0.85, y_max - size_y)
        rectangles.append((x0, y0, x0 + size_x, y0 + size_y))
    isolated_indices = list(range(isolated_start_idx, len(rectangles)))

    areas = np.array(
        [abs((r[2] - r[0]) * (r[3] - r[1])) for r in rectangles]
    )

    return {
        "rectangles": rectangles,
        "nested_indices": expected_nested,
        "nesting_groups": nesting_groups,
        "overlapping_indices": overlapping_indices,
        "isolated_indices": isolated_indices,
        "areas": areas,
    }