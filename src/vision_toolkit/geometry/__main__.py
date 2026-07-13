from __future__ import annotations

import numpy as np

from vision_toolkit.geometry.gjk_2d import gjk_nesterov_accelerated_2d
from vision_toolkit.geometry.gjk_3d import (
    polyhedron_intersect_capsule,
    polyhedron_intersect_polyhedron,
    polyhedron_intersect_sphere,
)
from vision_toolkit.geometry.supports import (
    box_support_2d,
    circle_support,
    polygon_support,
)


def run_2d_demo() -> None:
    box1 = box_support_2d(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    box2 = box_support_2d(np.array([1.5, 0.0]), np.array([1.0, 1.0]))
    intersect, dist, _, iters = gjk_nesterov_accelerated_2d(box1, box2)
    print(f"Box intersection: {intersect}, distance: {dist}, iterations: {iters}")

    circle1 = circle_support(np.array([0.0, 0.0]), 1.0)
    circle2 = circle_support(np.array([3.0, 0.0]), 1.0)
    intersect, dist, _, iters = gjk_nesterov_accelerated_2d(circle1, circle2)
    print(f"Circle intersection: {intersect}, distance: {dist}, iterations: {iters}")

    triangle = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
    poly1 = polygon_support(triangle)
    poly2 = polygon_support(triangle + np.array([0.5, 0.5]))
    intersect, dist, _, iters = gjk_nesterov_accelerated_2d(poly1, poly2)
    print(f"Triangle intersection: {intersect}, distance: {dist}, iterations: {iters}")


def run_3d_demo() -> None:
    cube = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float32,
    )

    result = polyhedron_intersect_sphere(
        cube, np.array([3.0, 0.0, 0.0], dtype=np.float32), 1.5
    )
    print(f"Cube-Sphere: hit={result.hit}, distance={result.distance:.4f}")

    cube2 = cube + np.array([2.5, 0, 0], dtype=np.float32)
    result2 = polyhedron_intersect_polyhedron(cube, cube2)
    print(f"Cube-Cube: hit={result2.hit}, distance={result2.distance:.4f}")

    result3 = polyhedron_intersect_capsule(
        cube,
        np.array([0, 3, 0], dtype=np.float32),
        np.array([0, 5, 0], dtype=np.float32),
        0.5,
    )
    print(f"Cube-Capsule: hit={result3.hit}, distance={result3.distance:.4f}")


if __name__ == "__main__":
    run_2d_demo()
    run_3d_demo()