from __future__ import annotations

import numba
import numpy as np

from vision_toolkit.geometry.settings import GJK2DSettings
from vision_toolkit.geometry.supports import SupportFn


def gjk_nesterov_accelerated_2d_intersection(
    support1: SupportFn,
    support2: SupportFn,
    settings: GJK2DSettings = GJK2DSettings(),
) -> bool:
    return gjk_nesterov_accelerated_2d(support1, support2, settings)[0]


def gjk_nesterov_accelerated_2d_distance(
    support1: SupportFn,
    support2: SupportFn,
    settings: GJK2DSettings = GJK2DSettings(),
) -> float:
    return max(gjk_nesterov_accelerated_2d(support1, support2, settings)[1], 0.0)


def gjk_nesterov_accelerated_2d(
    support1: SupportFn,
    support2: SupportFn,
    settings: GJK2DSettings = GJK2DSettings(),
) -> tuple[bool, float, np.ndarray, int]:
    use_acceleration = settings.use_nesterov_acceleration
    tolerance = settings.tolerance
    upper_bound = settings.upper_bound

    inflation = 0.0
    upper_bound += inflation
    alpha = 0.0
    inside = False
    simplex = np.zeros((3, 2), dtype=np.float64)
    simplex_len = 0
    distance = 0.0
    ray = np.array([1.0, 0.0])
    ray_len = 1.0
    ray_dir = ray.copy()
    support_point = ray.copy()

    i = 0
    while i < settings.max_iterations:
        if ray_len < tolerance:
            distance = -inflation
            inside = True
            break

        if use_acceleration:
            momentum = (i + 1) / (i + 3)
            y = momentum * ray + (1.0 - momentum) * support_point
            ray_dir = momentum * ray_dir + (1.0 - momentum) * y
        else:
            ray_dir = ray

        s0 = support1(-ray_dir)
        s1 = support2(ray_dir)
        simplex[simplex_len] = s0 - s1
        support_point = simplex[simplex_len]
        simplex_len += 1

        omega = ray_dir.dot(support_point) / np.linalg.norm(ray_dir)
        if omega > upper_bound:
            distance = omega - inflation
            inside = False
            break

        if use_acceleration:
            frank_wolfe_duality_gap = 2 * ray.dot(ray - support_point)
            if frank_wolfe_duality_gap - tolerance <= 0:
                use_acceleration = False
                simplex_len -= 1
                i += 1
                continue

        alpha = max(alpha, omega)
        diff = ray_len - alpha
        cv_check_passed = (diff - tolerance * ray_len) <= 0

        if i > 0 and cv_check_passed:
            simplex_len -= 1
            if use_acceleration:
                use_acceleration = False
                i += 1
                continue
            distance = ray_len - inflation
            inside = distance < tolerance
            break

        if simplex_len == 1:
            ray = support_point.copy()
        elif simplex_len == 2:
            ray, simplex_len, inside = project_line_origin_2d(simplex)
        else:
            ray, simplex_len, inside = project_triangle_origin_2d(simplex)

        if not inside:
            ray_len = np.linalg.norm(ray)
        if inside or ray_len == 0:
            distance = -inflation - 1.0
            inside = True
            break

        i += 1

    return inside, distance, simplex[:simplex_len], i


@numba.njit(cache=True)
def origin_to_point_2d(simplex, a):
    simplex[0] = a.copy()
    return a.copy(), 1


@numba.njit(cache=True)
def origin_to_segment_2d(simplex, a, b, ab, ab_dot_a0):
    ray = (ab.dot(b) * a + ab_dot_a0 * b) / ab.dot(ab)
    simplex[0] = b.copy()
    simplex[1] = a.copy()
    return ray, 2


@numba.njit(cache=True)
def origin_inside_triangle_2d(simplex, a, b, c):
    simplex[0] = c.copy()
    simplex[1] = b.copy()
    simplex[2] = a.copy()
    return np.zeros(2), 3, True


@numba.njit(cache=True)
def project_line_origin_2d(line):
    a = line[1]
    b = line[0]
    ab = b - a
    d = np.dot(ab, -a)

    if d == 0:
        ray, simplex_len = origin_to_point_2d(line, a)
        return ray, simplex_len, np.all(a == 0.0)
    if d < 0:
        ray, simplex_len = origin_to_point_2d(line, a)
    else:
        ray, simplex_len = origin_to_segment_2d(line, a, b, ab, d)
    return ray, simplex_len, False


@numba.njit(cache=True)
def triple_product_2d(a, b, c):
    ac = a.dot(c)
    bc = b.dot(c)
    return np.array([b[0] * ac - a[0] * bc, b[1] * ac - a[1] * bc])


@numba.njit(cache=True)
def t_b_2d(triangle, a, b, ab):
    towards_b = ab.dot(-a)
    if towards_b < 0:
        return origin_to_point_2d(triangle, a)
    return origin_to_segment_2d(triangle, a, b, ab, towards_b)


@numba.njit(cache=True)
def project_triangle_origin_2d(triangle):
    a = triangle[2]
    b = triangle[1]
    c = triangle[0]
    ab = b - a
    ac = c - a
    ao = -a
    ab_perp = triple_product_2d(ac, ab, ab)
    ac_perp = triple_product_2d(ab, ac, ac)

    if ab_perp.dot(ao) > 0:
        ray, simplex_len = t_b_2d(triangle, a, b, ab)
        return ray, simplex_len, False

    if ac_perp.dot(ao) > 0:
        towards_c = ac.dot(ao)
        if towards_c >= 0:
            ray, simplex_len = origin_to_segment_2d(triangle, a, c, ac, towards_c)
        else:
            ray, simplex_len = origin_to_point_2d(triangle, a)
        return ray, simplex_len, False

    return origin_inside_triangle_2d(triangle, a, b, c)