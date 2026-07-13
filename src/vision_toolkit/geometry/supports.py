from __future__ import annotations

from typing import Callable

import numpy as np

SupportFn = Callable[[np.ndarray], np.ndarray]


def circle_support(center: np.ndarray, radius: float) -> SupportFn:
    def support(direction: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return center + np.array([radius, 0.0])
        return center + radius * direction / norm

    return support


def polygon_support(vertices: np.ndarray) -> SupportFn:
    def support(direction: np.ndarray) -> np.ndarray:
        dots = vertices @ direction
        return vertices[np.argmax(dots)].copy()

    return support


def box_support_2d(center: np.ndarray, half_extents: np.ndarray) -> SupportFn:
    def support(direction: np.ndarray) -> np.ndarray:
        return center + np.sign(direction) * half_extents

    return support


def ellipse_support(center: np.ndarray, radii: np.ndarray) -> SupportFn:
    def support(direction: np.ndarray) -> np.ndarray:
        scaled = radii * direction
        norm = np.linalg.norm(scaled)
        if norm < 1e-10:
            return center + np.array([radii[0], 0.0])
        return center + radii * scaled / norm

    return support


def coco_bbox_support(bbox: list[float]) -> SupportFn:
    x, y, w, h = bbox
    center = np.array([x + w / 2.0, y + h / 2.0])
    half_extents = np.array([w / 2.0, h / 2.0])
    return box_support_2d(center, half_extents)


def coco_segmentation_support(segmentation: list[float]) -> SupportFn:
    vertices = np.asarray(segmentation, dtype=np.float64).reshape(-1, 2)
    return polygon_support(vertices)


def polyhedron_support(d: np.ndarray, verts: np.ndarray) -> tuple[np.ndarray, int]:
    dots = verts @ d
    imax = int(np.argmax(dots))
    return verts[imax].copy(), imax


def line_support(
    d: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, int]:
    if np.dot(a, d) < np.dot(b, d):
        return b.copy(), 1
    return a.copy(), 0