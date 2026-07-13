from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vision_toolkit.geometry.settings import GJK3DSettings
from vision_toolkit.geometry.supports import line_support, polyhedron_support


@dataclass
class GJKVertex:
    a: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    b: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    p: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    aid: int = 0
    bid: int = 0


@dataclass
class GJKSupport:
    aid: int = 0
    bid: int = 0
    a: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    b: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    da: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    db: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


@dataclass
class GJKSimplex:
    settings: GJK3DSettings = field(default_factory=GJK3DSettings)
    max_iter: int = 0
    iter: int = 0
    hit: bool = False
    cnt: int = 0
    v: list[GJKVertex] = field(
        default_factory=lambda: [GJKVertex() for _ in range(4)]
    )
    bc: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    D: float = np.finfo(np.float32).max


@dataclass
class GJKResult:
    hit: bool = False
    p0: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    p1: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    distance: float = 0.0
    iterations: int = 0


def f3box(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return np.dot(np.cross(a, b), c)


def inv_sqrt(n: float) -> float:
    if n <= 0:
        return 0.0
    return 1.0 / np.sqrt(n)


def gjk(s: GJKSimplex, sup: GJKSupport) -> bool:
    if s.max_iter > 0 and s.iter >= s.max_iter:
        return False

    if s.cnt == 0:
        s.D = np.finfo(np.float32).max
        s.max_iter = (
            s.settings.max_iterations if s.max_iter == 0 else s.max_iter
        )

    for i in range(s.cnt):
        if sup.aid == s.v[i].aid and sup.bid == s.v[i].bid:
            return False

    vert = s.v[s.cnt]
    vert.a[:] = sup.a
    vert.b[:] = sup.b
    vert.p[:] = sup.b - sup.a
    vert.aid = sup.aid
    vert.bid = sup.bid
    s.bc[s.cnt] = 1.0
    s.cnt += 1

    if s.cnt == 2:
        a, b = s.v[0].p.copy(), s.v[1].p.copy()
        ab, ba = a - b, b - a
        u, v = np.dot(b, ba), np.dot(a, ab)

        if v <= 0.0:
            s.bc[0] = 1.0
            s.cnt = 1
        elif u <= 0.0:
            s.v[0] = s.v[1]
            s.bc[0] = 1.0
            s.cnt = 1
        else:
            s.bc[0], s.bc[1] = u, v
            s.cnt = 2

    elif s.cnt == 3:
        a, b, c = s.v[0].p.copy(), s.v[1].p.copy(), s.v[2].p.copy()
        ab, ba = a - b, b - a
        bc, cb = b - c, c - b
        ca, ac = c - a, a - c

        u_ab, v_ab = np.dot(b, ba), np.dot(a, ab)
        u_bc, v_bc = np.dot(c, cb), np.dot(b, bc)
        u_ca, v_ca = np.dot(a, ac), np.dot(c, ca)

        if v_ab <= 0.0 and u_ca <= 0.0:
            s.bc[0] = 1.0
            s.cnt = 1
        elif u_ab <= 0.0 and v_bc <= 0.0:
            s.v[0] = s.v[1]
            s.bc[0] = 1.0
            s.cnt = 1
        elif u_bc <= 0.0 and v_ca <= 0.0:
            s.v[0] = s.v[2]
            s.bc[0] = 1.0
            s.cnt = 1
        else:
            n = np.cross(ba, ca)
            u_abc = np.dot(np.cross(b, c), n)
            v_abc = np.dot(np.cross(c, a), n)
            w_abc = np.dot(np.cross(a, b), n)

            if u_ab > 0.0 and v_ab > 0.0 and w_abc <= 0.0:
                s.bc[0], s.bc[1] = u_ab, v_ab
                s.cnt = 2
            elif u_bc > 0.0 and v_bc > 0.0 and u_abc <= 0.0:
                s.v[0], s.v[1] = s.v[1], s.v[2]
                s.bc[0], s.bc[1] = u_bc, v_bc
                s.cnt = 2
            elif u_ca > 0.0 and v_ca > 0.0 and v_abc <= 0.0:
                s.v[1], s.v[0] = s.v[0], s.v[2]
                s.bc[0], s.bc[1] = u_ca, v_ca
                s.cnt = 2
            else:
                s.bc[0], s.bc[1], s.bc[2] = u_abc, v_abc, w_abc
                s.cnt = 3

    elif s.cnt == 4:
        a, b, c, d = [s.v[i].p.copy() for i in range(4)]
        ab, ba = a - b, b - a
        bc, cb = b - c, c - b
        ca, ac = c - a, a - c
        db, bd = d - b, b - d
        dc, cd = d - c, c - d
        da, ad = d - a, a - d

        u_ab, v_ab = np.dot(b, ba), np.dot(a, ab)
        u_bc, v_bc = np.dot(c, cb), np.dot(b, bc)
        u_ca, v_ca = np.dot(a, ac), np.dot(c, ca)
        u_bd, v_bd = np.dot(d, db), np.dot(b, bd)
        u_dc, v_dc = np.dot(c, cd), np.dot(d, dc)
        u_ad, v_ad = np.dot(d, da), np.dot(a, ad)

        if v_ab <= 0.0 and u_ca <= 0.0 and v_ad <= 0.0:
            s.bc[0] = 1.0
            s.cnt = 1
        elif u_ab <= 0.0 and v_bc <= 0.0 and v_bd <= 0.0:
            s.v[0] = s.v[1]
            s.bc[0] = 1.0
            s.cnt = 1
        elif u_bc <= 0.0 and v_ca <= 0.0 and u_dc <= 0.0:
            s.v[0] = s.v[2]
            s.bc[0] = 1.0
            s.cnt = 1
        elif u_bd <= 0.0 and v_dc <= 0.0 and u_ad <= 0.0:
            s.v[0] = s.v[3]
            s.bc[0] = 1.0
            s.cnt = 1
        else:
            n = np.cross(da, ba)
            u_adb = np.dot(np.cross(d, b), n)
            v_adb = np.dot(np.cross(b, a), n)
            w_adb = np.dot(np.cross(a, d), n)

            n = np.cross(ca, da)
            u_acd = np.dot(np.cross(c, d), n)
            v_acd = np.dot(np.cross(d, a), n)
            w_acd = np.dot(np.cross(a, c), n)

            n = np.cross(bc, dc)
            u_cbd = np.dot(np.cross(b, d), n)
            v_cbd = np.dot(np.cross(d, c), n)
            w_cbd = np.dot(np.cross(c, b), n)

            n = np.cross(ba, ca)
            u_abc = np.dot(np.cross(b, c), n)
            v_abc = np.dot(np.cross(c, a), n)
            w_abc = np.dot(np.cross(a, b), n)

            if w_abc <= 0.0 and v_adb <= 0.0 and u_ab > 0.0 and v_ab > 0.0:
                s.bc[0], s.bc[1] = u_ab, v_ab
                s.cnt = 2
            elif u_abc <= 0.0 and w_cbd <= 0.0 and u_bc > 0.0 and v_bc > 0.0:
                s.v[0], s.v[1] = s.v[1], s.v[2]
                s.bc[0], s.bc[1] = u_bc, v_bc
                s.cnt = 2
            elif v_abc <= 0.0 and w_acd <= 0.0 and u_ca > 0.0 and v_ca > 0.0:
                s.v[1], s.v[0] = s.v[0], s.v[2]
                s.bc[0], s.bc[1] = u_ca, v_ca
                s.cnt = 2
            elif v_cbd <= 0.0 and u_acd <= 0.0 and u_dc > 0.0 and v_dc > 0.0:
                s.v[0], s.v[1] = s.v[3], s.v[2]
                s.bc[0], s.bc[1] = u_dc, v_dc
                s.cnt = 2
            elif v_acd <= 0.0 and w_adb <= 0.0 and u_ad > 0.0 and v_ad > 0.0:
                s.v[1] = s.v[3]
                s.bc[0], s.bc[1] = u_ad, v_ad
                s.cnt = 2
            elif u_cbd <= 0.0 and u_adb <= 0.0 and u_bd > 0.0 and v_bd > 0.0:
                s.v[0], s.v[1] = s.v[1], s.v[3]
                s.bc[0], s.bc[1] = u_bd, v_bd
                s.cnt = 2
            else:
                denom = f3box(cb, ab, db)
                volume = 1.0 if denom == 0 else 1.0 / denom
                u_abcd = f3box(c, d, b) * volume
                v_abcd = f3box(c, a, d) * volume
                w_abcd = f3box(d, a, b) * volume
                x_abcd = f3box(b, a, c) * volume

                if x_abcd < 0.0 and u_abc > 0.0 and v_abc > 0.0 and w_abc > 0.0:
                    s.bc[0], s.bc[1], s.bc[2] = u_abc, v_abc, w_abc
                    s.cnt = 3
                elif u_abcd < 0.0 and u_cbd > 0.0 and v_cbd > 0.0 and w_cbd > 0.0:
                    s.v[0], s.v[2] = s.v[2], s.v[3]
                    s.bc[0], s.bc[1], s.bc[2] = u_cbd, v_cbd, w_cbd
                    s.cnt = 3
                elif v_abcd < 0.0 and u_acd > 0.0 and v_acd > 0.0 and w_acd > 0.0:
                    s.v[1], s.v[2] = s.v[2], s.v[3]
                    s.bc[0], s.bc[1], s.bc[2] = u_acd, v_acd, w_acd
                    s.cnt = 3
                elif w_abcd < 0.0 and u_adb > 0.0 and v_adb > 0.0 and w_adb > 0.0:
                    s.v[2], s.v[1] = s.v[1], s.v[3]
                    s.bc[0], s.bc[1], s.bc[2] = u_adb, v_adb, w_adb
                    s.cnt = 3
                else:
                    s.bc[0], s.bc[1], s.bc[2], s.bc[3] = (
                        u_abcd,
                        v_abcd,
                        w_abcd,
                        x_abcd,
                    )
                    s.cnt = 4

    if s.cnt == 4:
        s.hit = True
        return False

    denom = sum(s.bc[: s.cnt])
    denom = 1.0 / denom if denom != 0 else 1.0

    if s.cnt == 1:
        pnt = s.v[0].p.copy()
    elif s.cnt == 2:
        pnt = s.v[0].p * (denom * s.bc[0]) + s.v[1].p * (denom * s.bc[1])
    else:
        pnt = (
            s.v[0].p * (denom * s.bc[0])
            + s.v[1].p * (denom * s.bc[1])
            + s.v[2].p * (denom * s.bc[2])
        )

    d2 = np.dot(pnt, pnt)
    if d2 >= s.D:
        return False
    s.D = d2

    if s.cnt == 1:
        d = -s.v[0].p
    elif s.cnt == 2:
        ba = s.v[1].p - s.v[0].p
        b0 = -s.v[1].p
        t = np.cross(ba, b0)
        d = np.cross(t, ba)
    else:
        ab = s.v[1].p - s.v[0].p
        ac = s.v[2].p - s.v[0].p
        n = np.cross(ab, ac)
        d = n if np.dot(n, s.v[0].p) <= 0.0 else -n

    if np.dot(d, d) < s.settings.epsilon * s.settings.epsilon:
        return False

    sup.da[:] = -d
    sup.db[:] = d
    s.iter += 1
    return True


def gjk_analyze(s: GJKSimplex) -> GJKResult:
    res = GJKResult()
    res.iterations = s.iter
    res.hit = s.hit

    denom = sum(s.bc[: s.cnt])
    denom = 1.0 / denom if denom != 0 else 1.0

    if s.cnt == 1:
        res.p0[:] = s.v[0].a
        res.p1[:] = s.v[0].b
    elif s.cnt == 2:
        res.p0[:] = s.v[0].a * (denom * s.bc[0]) + s.v[1].a * (denom * s.bc[1])
        res.p1[:] = s.v[0].b * (denom * s.bc[0]) + s.v[1].b * (denom * s.bc[1])
    elif s.cnt == 3:
        res.p0[:] = (
            s.v[0].a * (denom * s.bc[0])
            + s.v[1].a * (denom * s.bc[1])
            + s.v[2].a * (denom * s.bc[2])
        )
        res.p1[:] = (
            s.v[0].b * (denom * s.bc[0])
            + s.v[1].b * (denom * s.bc[1])
            + s.v[2].b * (denom * s.bc[2])
        )
    elif s.cnt == 4:
        res.p0[:] = (
            s.v[0].a * (denom * s.bc[0])
            + s.v[1].a * (denom * s.bc[1])
            + s.v[2].a * (denom * s.bc[2])
            + s.v[3].a * (denom * s.bc[3])
        )
        res.p1[:] = res.p0

    if not res.hit:
        d = res.p1 - res.p0
        res.distance = np.sqrt(np.dot(d, d))
    else:
        res.distance = 0.0

    return res


def gjk_quad(res: GJKResult, a_radius: float, b_radius: float) -> None:
    epsilon = GJK3DSettings().epsilon
    radius = a_radius + b_radius
    if res.distance > epsilon and res.distance > radius:
        res.distance -= radius
        n = res.p1 - res.p0
        l2 = np.dot(n, n)
        if l2 != 0.0:
            n *= inv_sqrt(l2)
        res.p0 += n * a_radius
        res.p1 -= n * b_radius
    else:
        p = res.p0 + res.p1
        res.p0[:] = p * 0.5
        res.p1[:] = res.p0
        res.distance = 0.0
        res.hit = True


def polyhedron_intersect_sphere(
    verts: np.ndarray,
    center: np.ndarray,
    radius: float,
    settings: GJK3DSettings = GJK3DSettings(),
) -> GJKResult:
    sup = GJKSupport()
    sup.a[:] = verts[0]
    sup.b[:] = center

    gsx = GJKSimplex(settings=settings)
    while gjk(gsx, sup):
        sup.a, sup.aid = polyhedron_support(sup.da, verts)

    res = gjk_analyze(gsx)
    gjk_quad(res, 0, radius)
    return res


def polyhedron_intersect_capsule(
    verts: np.ndarray,
    ca: np.ndarray,
    cb: np.ndarray,
    cr: float,
    settings: GJK3DSettings = GJK3DSettings(),
) -> GJKResult:
    sup = GJKSupport()
    sup.a[:] = verts[0]
    sup.b[:] = ca

    gsx = GJKSimplex(settings=settings)
    while gjk(gsx, sup):
        sup.a, sup.aid = polyhedron_support(sup.da, verts)
        sup.b, sup.bid = line_support(sup.db, ca, cb)

    res = gjk_analyze(gsx)
    gjk_quad(res, 0, cr)
    return res


def polyhedron_intersect_polyhedron(
    verts_a: np.ndarray,
    verts_b: np.ndarray,
    settings: GJK3DSettings = GJK3DSettings(),
) -> GJKResult:
    sup = GJKSupport()
    sup.a[:] = verts_a[0]
    sup.b[:] = verts_b[0]

    gsx = GJKSimplex(settings=settings)
    while gjk(gsx, sup):
        sup.a, sup.aid = polyhedron_support(sup.da, verts_a)
        sup.b, sup.bid = polyhedron_support(sup.db, verts_b)

    return gjk_analyze(gsx)


def transform_point(v: np.ndarray, rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return v @ rot + pos


def transform_direction_inverse(
    d: np.ndarray, rot: np.ndarray, pos: np.ndarray
) -> np.ndarray:
    return (d - pos) @ rot.T


def polyhedron_intersect_polyhedron_transformed(
    verts_a: np.ndarray,
    pos_a: np.ndarray,
    rot_a: np.ndarray,
    verts_b: np.ndarray,
    pos_b: np.ndarray,
    rot_b: np.ndarray,
    settings: GJK3DSettings = GJK3DSettings(),
) -> GJKResult:
    sup = GJKSupport()
    sup.a[:] = transform_point(verts_a[0], rot_a, pos_a)
    sup.b[:] = transform_point(verts_b[0], rot_b, pos_b)

    gsx = GJKSimplex(settings=settings)
    while gjk(gsx, sup):
        da = transform_direction_inverse(sup.da, rot_a, pos_a)
        db = transform_direction_inverse(sup.db, rot_b, pos_b)

        sa, sup.aid = polyhedron_support(da, verts_a)
        sb, sup.bid = polyhedron_support(db, verts_b)

        sup.a[:] = transform_point(sa, rot_a, pos_a)
        sup.b[:] = transform_point(sb, rot_b, pos_b)

    return gjk_analyze(gsx)