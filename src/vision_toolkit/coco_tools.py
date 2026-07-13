from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, fields
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Any, Optional, Union

import mmh3
import numpy as np
import yaml
from bitarray import bitarray
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from sortedcontainers import SortedList

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def hydrate(cls: type, mapping: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in mapping.items() if k in known}
    return cls(**kwargs)


@dataclass(frozen=True)
class IntersectionSettings:
    epsilon: float = 1e-9
    min_vertices: int = 4


@dataclass(frozen=True)
class BloomSettings:
    false_positive_rate: float = 0.001
    capacity_multiplier: float = 1.3
    expected_growth_ratio: float = 0.5
    use_double_hashing: bool = True


@dataclass(frozen=True)
class SliceSettings:
    tile_width: int
    tile_height: int
    ann_file_path: str


@dataclass(frozen=True)
class SplitterSettings:
    input_json_path: str
    output_dir: str
    json_indent: int = 4
    filename_separator: str = "_"
    resolution_separator: str = "x"


class AnnotationBloomFilter:

    def __init__(self, capacity: int, settings: BloomSettings = BloomSettings()):
        self._validate_capacity(capacity)
        self._validate_fp_rate(settings.false_positive_rate)
        self.capacity = capacity
        self.fp_rate = settings.false_positive_rate
        self.use_double_hashing = settings.use_double_hashing
        self.m = self._calculate_size(capacity, self.fp_rate)
        self.k = self._calculate_hash_count(self.m, capacity)
        self.bit_array = bitarray(self.m)
        self.bit_array.setall(0)
        self.items_count = 0

    def _validate_capacity(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

    def _validate_fp_rate(self, rate: float) -> None:
        is_invalid = rate <= 0 or rate >= 1
        if is_invalid:
            raise ValueError(f"False positive rate must be in (0,1), got {rate}")

    @staticmethod
    def _calculate_size(n: int, p: float) -> int:
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def _calculate_hash_count(m: int, n: int) -> int:
        k = (m / n) * math.log(2)
        return max(1, int(math.ceil(k)))

    def _positions_double_hash(self, item_str: str) -> list[int]:
        h1 = mmh3.hash(item_str, seed=0)
        h2 = mmh3.hash(item_str, seed=1)
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def _positions_multi_hash(self, item_str: str) -> list[int]:
        return [mmh3.hash(item_str, seed=i) % self.m for i in range(self.k)]

    def _positions(self, item: Any) -> list[int]:
        item_str = str(item)
        if self.use_double_hashing:
            return self._positions_double_hash(item_str)
        return self._positions_multi_hash(item_str)

    def add(self, item: Any) -> bool:
        positions = self._positions(item)
        is_new = any(self.bit_array[pos] == 0 for pos in positions)
        for pos in positions:
            self.bit_array[pos] = 1
        if is_new:
            self.items_count += 1
        return is_new

    def might_exist(self, item: Any) -> bool:
        positions = self._positions(item)
        return all(self.bit_array[pos] == 1 for pos in positions)

    def definitely_new(self, item: Any) -> bool:
        return not self.might_exist(item)

    def __contains__(self, item: Any) -> bool:
        return self.might_exist(item)

    def get_stats(self) -> dict[str, Any]:
        bits_set = self.bit_array.count()
        fill_ratio = bits_set / self.m if self.m > 0 else 0.0
        return {
            "capacity": self.capacity,
            "items_added": self.items_count,
            "utilization": self.items_count / self.capacity if self.capacity else 0.0,
            "bit_array_size": self.m,
            "bits_set": bits_set,
            "fill_ratio": fill_ratio,
            "hash_functions": self.k,
            "target_fp_rate": self.fp_rate,
            "actual_fp_rate": fill_ratio ** self.k,
            "memory_bytes": self.m // 8,
            "memory_mb": (self.m // 8) / (1024 * 1024),
            "bits_per_item": self.m / self.capacity if self.capacity else 0.0,
        }

    def should_resize(self, threshold: float = 0.7) -> bool:
        return self.items_count >= self.capacity * threshold

    def export_state(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "false_positive_rate": self.fp_rate,
            "use_double_hashing": self.use_double_hashing,
            "items_count": self.items_count,
            "bit_array": self.bit_array.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "AnnotationBloomFilter":
        settings = BloomSettings(
            false_positive_rate=state["false_positive_rate"],
            use_double_hashing=state["use_double_hashing"],
        )
        bf = cls(capacity=state["capacity"], settings=settings)
        bf.items_count = state["items_count"]
        bf.bit_array = bitarray(state["bit_array"])
        return bf


def calculate_bloom_parameters(
    num_annotations: int,
    false_positive_rate: float = 0.001,
) -> dict[str, Any]:
    m = AnnotationBloomFilter._calculate_size(num_annotations, false_positive_rate)
    k = AnnotationBloomFilter._calculate_hash_count(m, num_annotations)
    return {
        "num_annotations": num_annotations,
        "false_positive_rate": false_positive_rate,
        "bit_array_size": m,
        "hash_functions": k,
        "memory_bytes": m // 8,
        "memory_mb": (m // 8) / (1024 * 1024),
        "bits_per_annotation": m / num_annotations,
    }


@dataclass(frozen=True)
class PolygonGenerator:
    num_vertices: int
    min_radius: float
    max_radius: float
    center: tuple[float, float]
    angle_variation: float

    def generate_polygon(self) -> np.ndarray:
        base_angles = np.linspace(0, 2 * np.pi, self.num_vertices, endpoint=False)
        radii = np.random.uniform(self.min_radius, self.max_radius, self.num_vertices)
        angle_var_abs = self.angle_variation * (2 * np.pi / self.num_vertices)
        offsets = np.random.uniform(-angle_var_abs, angle_var_abs, self.num_vertices)
        angles = base_angles + offsets
        x_coords = radii * np.cos(angles) + self.center[0]
        y_coords = radii * np.sin(angles) + self.center[1]
        return np.stack((x_coords, y_coords), axis=1)

    @staticmethod
    def bounding_box(points: np.ndarray) -> list[float]:
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    @staticmethod
    def area(points: np.ndarray) -> float:
        if points.shape[0] < 3:
            return 0.0
        x, y = points[:, 0], points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class EventType(Enum):
    LEFT = 0
    RIGHT = 1


@dataclass(order=True)
class SweepEvent:
    x: float
    event_type: EventType = field(compare=False)
    segment_idx: int = field(compare=False)
    point: tuple[float, float] = field(compare=False)


@dataclass
class Segment:
    p1: tuple[float, float]
    p2: tuple[float, float]
    idx: int

    def __post_init__(self):
        needs_swap = self.p1[0] > self.p2[0] or (
            self.p1[0] == self.p2[0] and self.p1[1] > self.p2[1]
        )
        if needs_swap:
            self.p1, self.p2 = self.p2, self.p1

    def y_at_x(self, x: float) -> float:
        if self.p1[0] == self.p2[0]:
            return self.p1[1]
        t = (x - self.p1[0]) / (self.p2[0] - self.p1[0])
        return self.p1[1] + t * (self.p2[1] - self.p1[1])


@dataclass(frozen=True)
class IntersectionResult:
    is_intersecting: bool
    point: Optional[tuple[float, float]] = None


@dataclass
class ProcessingState:
    status: "SweepLineStatus"
    segments: list[Segment]
    n_segments: int
    processed: set
    result: Optional[tuple[float, float]] = None


class SweepLineStatus:

    def __init__(self, eps: float):
        self.eps = eps
        self.current_x = 0.0
        self._segments: dict[int, Segment] = {}
        self._active: SortedList = SortedList(
            key=lambda idx: self._segments[idx].y_at_x(self.current_x)
        )

    def set_x(self, x: float) -> None:
        self.current_x = x

    def insert(self, segment: Segment) -> None:
        self._segments[segment.idx] = segment
        self._active.add(segment.idx)

    def remove(self, segment_idx: int) -> None:
        if segment_idx in self._active:
            self._active.remove(segment_idx)

    def get_neighbors(self, segment_idx: int) -> tuple[Optional[int], Optional[int]]:
        if segment_idx not in self._active:
            return None, None
        pos = self._active.index(segment_idx)
        below = self._active[pos - 1] if pos > 0 else None
        above = self._active[pos + 1] if pos < len(self._active) - 1 else None
        return below, above


class PolygonIntersectionChecker:

    def __init__(self, settings: IntersectionSettings = IntersectionSettings()):
        self.eps = settings.epsilon
        self.min_vertices = settings.min_vertices

    def _ccw(self, a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def _segments_intersect(
        self, s1: Segment, s2: Segment
    ) -> Optional[tuple[float, float]]:
        d1 = self._ccw(s2.p1, s2.p2, s1.p1)
        d2 = self._ccw(s2.p1, s2.p2, s1.p2)
        d3 = self._ccw(s1.p1, s1.p2, s2.p1)
        d4 = self._ccw(s1.p1, s1.p2, s2.p2)
        cross1 = (d1 > self.eps and d2 < -self.eps) or (d1 < -self.eps and d2 > self.eps)
        cross2 = (d3 > self.eps and d4 < -self.eps) or (d3 < -self.eps and d4 > self.eps)
        if cross1 and cross2:
            t = d1 / (d1 - d2)
            ix = s1.p1[0] + t * (s1.p2[0] - s1.p1[0])
            iy = s1.p1[1] + t * (s1.p2[1] - s1.p1[1])
            return (ix, iy)
        return None

    def _build_segments(self, points: np.ndarray) -> list[Segment]:
        n = len(points)
        indices = np.arange(n)
        next_indices = (indices + 1) % n
        return [
            Segment(tuple(points[i]), tuple(points[next_indices[i]]), i)
            for i in indices
        ]

    def _build_events(self, segments: list[Segment]) -> SortedList:
        events = SortedList()
        for seg in segments:
            events.add(SweepEvent(seg.p1[0], EventType.LEFT, seg.idx, seg.p1))
            events.add(SweepEvent(seg.p2[0], EventType.RIGHT, seg.idx, seg.p2))
        return events

    def _are_adjacent(self, idx1: int, idx2: int, n_segments: int) -> bool:
        diff = abs(idx1 - idx2)
        return diff == 1 or diff == n_segments - 1

    def _check_neighbor_intersection(
        self,
        seg: Segment,
        neighbor_idx: Optional[int],
        segments: list[Segment],
        n_segments: int,
        current_x: float,
    ) -> Optional[tuple[float, float]]:
        if neighbor_idx is None:
            return None
        if self._are_adjacent(seg.idx, neighbor_idx, n_segments):
            return None
        intersection = self._segments_intersect(seg, segments[neighbor_idx])
        if intersection is None:
            return None
        if intersection[0] < current_x - self.eps:
            return None
        return intersection

    def _process_left_event(
        self, event: SweepEvent, state: ProcessingState
    ) -> Optional[tuple[float, float]]:
        seg = state.segments[event.segment_idx]
        state.status.insert(seg)
        below, above = state.status.get_neighbors(event.segment_idx)
        below_hit = self._check_neighbor_intersection(
            seg, below, state.segments, state.n_segments, event.x
        )
        if below_hit:
            return below_hit
        return self._check_neighbor_intersection(
            seg, above, state.segments, state.n_segments, event.x
        )

    def _process_right_event(
        self, event: SweepEvent, state: ProcessingState
    ) -> Optional[tuple[float, float]]:
        below, above = state.status.get_neighbors(event.segment_idx)
        state.status.remove(event.segment_idx)
        if below is None or above is None:
            return None
        if self._are_adjacent(below, above, state.n_segments):
            return None
        intersection = self._segments_intersect(
            state.segments[below], state.segments[above]
        )
        if intersection is None:
            return None
        pair = tuple(sorted([below, above]))
        if pair in state.processed:
            return None
        state.processed.add(pair)
        if intersection[0] < event.x - self.eps:
            return None
        return intersection

    def _process_event(
        self, state: ProcessingState, event: SweepEvent
    ) -> ProcessingState:
        if state.result is not None:
            return state
        state.status.set_x(event.x)
        handlers = {
            EventType.LEFT: lambda: self._process_left_event(event, state),
            EventType.RIGHT: lambda: self._process_right_event(event, state),
        }
        handler = handlers.get(event.event_type)
        state.result = handler() if handler else None
        return state

    def check(self, points: Union[list[float], np.ndarray]) -> IntersectionResult:
        points_arr = np.asarray(points, dtype=np.float64)
        if points_arr.ndim == 1:
            if points_arr.size % 2 != 0:
                raise ValueError("Flat list must have even number of elements.")
            points_arr = points_arr.reshape(-1, 2)
        if len(points_arr) < self.min_vertices:
            return IntersectionResult(False, None)
        segments = self._build_segments(points_arr)
        events = self._build_events(segments)
        state = ProcessingState(
            status=SweepLineStatus(self.eps),
            segments=segments,
            n_segments=len(segments),
            processed=set(),
        )
        final_state = reduce(self._process_event, events, state)
        return IntersectionResult(final_state.result is not None, final_state.result)


def is_self_intersecting(
    points: Union[list[float], np.ndarray],
    settings: IntersectionSettings = IntersectionSettings(),
) -> bool:
    return PolygonIntersectionChecker(settings).check(points).is_intersecting


def get_intersection_point(
    points: Union[list[float], np.ndarray],
    settings: IntersectionSettings = IntersectionSettings(),
) -> Optional[tuple[float, float]]:
    return PolygonIntersectionChecker(settings).check(points).point


class SegmentTreeBase:

    def __init__(self, interval_lengths: list, y_coords: list):
        if not interval_lengths:
            raise ValueError("Interval list must not be empty.")
        self.N = 1
        while self.N < len(interval_lengths):
            self.N *= 2
        self.c = [0] * (2 * self.N)
        self.s = [0] * (2 * self.N)
        self.w = [0] * (2 * self.N)
        self.y_coords = y_coords
        for i, val in enumerate(interval_lengths):
            self.w[self.N + i] = val
        for p in range(self.N - 1, 0, -1):
            self.w[p] = self.w[2 * p] + self.w[2 * p + 1]

    def query_coverage(self):
        return self.s[1]

    def _change(self, p, start, span, i, k, offset):
        if start + span <= i or k <= start:
            return
        if i <= start and start + span <= k:
            self.c[p] += offset
        else:
            mid = span // 2
            self._change(2 * p, start, mid, i, k, offset)
            self._change(2 * p + 1, start + mid, mid, i, k, offset)
        self._update_s(p)

    def _update_s(self, p):
        if self.c[p] != 0:
            self.s[p] = self.w[p]
            return
        if p >= self.N:
            self.s[p] = 0
            return
        self.s[p] = self.s[2 * p] + self.s[2 * p + 1]


class CoverageTracker(SegmentTreeBase):

    def __init__(self, interval_lengths: list, y_coords: list):
        super().__init__(interval_lengths, y_coords)
        self.sweep_events = []
        self.active_rectangles: dict[int, set] = {}

    def modify_interval(self, i, k, offset, x_coord, rect_idx):
        for y in range(i, k):
            active = self.active_rectangles.setdefault(y, set())
            if offset == 1:
                active.add(rect_idx)
            else:
                active.discard(rect_idx)
            if active:
                self.sweep_events.append((x_coord, y, set(active)))
        self._change(1, 0, self.N, i, k, offset)


class DualListOverlapTracker(SegmentTreeBase):

    def __init__(self, interval_lengths: list, y_coords: list):
        super().__init__(interval_lengths, y_coords)
        self.overlaps = []
        self.sweep_events = []
        self.active_a: dict[int, set] = {}
        self.active_b: dict[int, set] = {}

    def modify_interval(self, i, k, offset, x_coord, rect_idx, is_from_list_a):
        for y in range(i, k):
            self._ensure_y_initialized(y)
            was_overlapping = bool(self.active_a[y]) and bool(self.active_b[y])
            old_sets = (set(self.active_a[y]), set(self.active_b[y]))
            self._update_target_set(y, is_from_list_a, offset, rect_idx)
            is_overlapping = bool(self.active_a[y]) and bool(self.active_b[y])
            new_sets = (set(self.active_a[y]), set(self.active_b[y]))
            self._maybe_append_sweep_event(
                was_overlapping, is_overlapping, old_sets, new_sets, x_coord, y
            )
        self._change(1, 0, self.N, i, k, offset)

    def _ensure_y_initialized(self, y):
        if y in self.active_a:
            return
        self.active_a[y] = set()
        self.active_b[y] = set()

    def _update_target_set(self, y, is_from_list_a, offset, rect_idx):
        target = self.active_a[y] if is_from_list_a else self.active_b[y]
        if offset == 1:
            target.add(rect_idx)
            return
        target.discard(rect_idx)

    def _maybe_append_sweep_event(
        self, was_overlapping, is_overlapping, old_sets, new_sets, x_coord, y
    ):
        changed_state = was_overlapping != is_overlapping
        sets_differ = is_overlapping and old_sets != new_sets
        if changed_state or sets_differ:
            self.sweep_events.append(
                (x_coord, y, set(self.active_a[y]), set(self.active_b[y]))
            )

    def find_overlaps_between_lists(self):
        self.sweep_events.sort()
        active_regions = {}
        for x, y, rects_a, rects_b in self.sweep_events:
            self._process_existing_region(active_regions, x, y)
            self._maybe_start_new_region(active_regions, y, x, rects_a, rects_b)

    def _process_existing_region(self, active_regions, x, y):
        if y not in active_regions:
            return
        start_x, start_rects_a, start_rects_b = active_regions.pop(y)
        if x <= start_x:
            return
        y_low = self.y_coords[y]
        y_high = self.y_coords[y + 1]
        self.overlaps.append((start_x, x, y_low, y_high, start_rects_a, start_rects_b))

    def _maybe_start_new_region(self, active_regions, y, x, rects_a, rects_b):
        if rects_a and rects_b:
            active_regions[y] = (x, rects_a, rects_b)


@dataclass(frozen=True)
class RectEvent:
    x: float
    rectangle: tuple
    is_start: bool
    rect_idx: int
    is_from_list_a: bool = True


class SweeplineSliceGenerator:

    def __init__(self, width: int, height: int, x_overlap: float, y_overlap: float):
        self.width = width
        self.height = height
        self.x_overlap = max(0.0, min(x_overlap, 0.99))
        self.y_overlap = max(0.0, min(y_overlap, 0.99))

    def generate_slices(
        self, target_tile_size: Optional[tuple[int, int]] = None
    ) -> list[list[int]]:
        tile_width, tile_height = self._resolve_tile_size(target_tile_size)
        return self._sweep_line_placement(tile_width, tile_height)

    def _resolve_tile_size(
        self, target_tile_size: Optional[tuple[int, int]]
    ) -> tuple[int, int]:
        if target_tile_size is None:
            num_tiles_x = max(2, int(1 / (1 - self.x_overlap)))
            num_tiles_y = max(2, int(1 / (1 - self.y_overlap)))
            tile_width = int(self.width / (num_tiles_x * (1 - self.x_overlap)))
            tile_height = int(self.height / (num_tiles_y * (1 - self.y_overlap)))
        else:
            tile_width, tile_height = target_tile_size
        return min(tile_width, self.width), min(tile_height, self.height)

    def _build_tile_positions(
        self, tile_width: int, tile_height: int
    ) -> list[tuple[int, int, int, int]]:
        stride_x = max(1, int(tile_width * (1 - self.x_overlap)))
        stride_y = max(1, int(tile_height * (1 - self.y_overlap)))
        positions = []
        y = 0
        while y < self.height:
            self._fill_row(positions, y, tile_width, tile_height, stride_x)
            y += stride_y
            reached_bottom_edge = (
                y + tile_height >= self.height and y < self.height - stride_y
            )
            if reached_bottom_edge:
                y1 = max(0, self.height - tile_height)
                if y1 < self.height:
                    self._fill_row(positions, y1, tile_width, tile_height, stride_x)
                break
        return positions

    def _fill_row(self, positions, y1, tile_width, tile_height, stride_x) -> None:
        x = 0
        while x < self.width:
            x2 = min(x + tile_width, self.width)
            y2 = min(y1 + tile_height, self.height)
            if x2 > x and y2 > y1:
                positions.append((x, y1, x2, y2))
            x += stride_x
            reached_right_edge = (
                x + tile_width >= self.width and x < self.width - stride_x
            )
            if reached_right_edge:
                edge_x1 = max(0, self.width - tile_width)
                if edge_x1 < self.width:
                    positions.append(
                        (edge_x1, y1, min(self.width, edge_x1 + tile_width), y2)
                    )
                break

    def _sweep_line_placement(
        self, tile_width: int, tile_height: int
    ) -> list[list[int]]:
        tile_positions = self._build_tile_positions(tile_width, tile_height)
        events = []
        for i, rect in enumerate(tile_positions):
            events.append(RectEvent(rect[0], rect, True, i))
            events.append(RectEvent(rect[2], rect, False, i))
        events.sort(key=lambda e: (e.x, not e.is_start))

        y_coords = sorted({y for rect in tile_positions for y in (rect[1], rect[3])})
        if len(y_coords) < 2:
            return [[0, 0, self.width, self.height]]

        y_intervals = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]
        y_mapping = {val: idx for idx, val in enumerate(y_coords)}
        tracker = CoverageTracker(y_intervals, y_coords)

        validated_tiles = set()
        for event in events:
            y0 = y_mapping.get(event.rectangle[1], 0)
            y1 = y_mapping.get(event.rectangle[3], 0)
            if y0 >= y1:
                continue
            offset = 1 if event.is_start else -1
            tracker.modify_interval(y0, y1, offset, event.x, event.rect_idx)
            if event.is_start:
                validated_tiles.add(event.rect_idx)

        slices = [list(tile_positions[idx]) for idx in sorted(validated_tiles)]
        return _deduplicate_slices(slices)


def _deduplicate_slices(slices: list[list[int]]) -> list[list[int]]:
    unique_slices = []
    seen = set()
    for s in slices:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_slices.append(s)
    return unique_slices


def generate_overlapping_slices_sweepline(
    resolution_to_overlap: dict[tuple[int, int], tuple[float, float]],
    target_tile_size: Optional[tuple[int, int]] = None,
) -> dict[tuple[int, int], list[list[int]]]:
    result: dict[tuple[int, int], list[list[int]]] = {}
    for (width, height), (x_overlap, y_overlap) in resolution_to_overlap.items():
        generator = SweeplineSliceGenerator(width, height, x_overlap, y_overlap)
        slices = generator.generate_slices(target_tile_size)
        result[(width, height)] = _deduplicate_slices(slices)
    return result


def _is_empty(collection) -> bool:
    return len(collection) == 0


def _normalize_rect(r) -> tuple:
    return (min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3]))


def _create_rect_events(normalized_a, normalized_b) -> list[RectEvent]:
    events = []
    for i, r in enumerate(normalized_a):
        if r[0] < r[2]:
            events.append(RectEvent(r[0], r, True, i, True))
            events.append(RectEvent(r[2], r, False, i, True))
    for i, r in enumerate(normalized_b):
        if r[0] < r[2]:
            events.append(RectEvent(r[0], r, True, i, False))
            events.append(RectEvent(r[2], r, False, i, False))
    return events


def find_overlapping_between_lists(rectangles_a, rectangles_b):
    if _is_empty(rectangles_a) or _is_empty(rectangles_b):
        return None, None, []

    normalized_a = [_normalize_rect(r) for r in rectangles_a]
    normalized_b = [_normalize_rect(r) for r in rectangles_b]
    events = _create_rect_events(normalized_a, normalized_b)
    if _is_empty(events):
        return None, None, []
    events.sort(key=lambda e: (e.x, not e.is_start))

    y_coords = sorted({y for r in normalized_a + normalized_b for y in (r[1], r[3])})
    if len(y_coords) < 2:
        return None, None, []

    y_intervals = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]
    y_mapping = {val: idx for idx, val in enumerate(y_coords)}
    tracker = DualListOverlapTracker(y_intervals, y_coords)

    for event in events:
        y0 = y_mapping[event.rectangle[1]]
        y1 = y_mapping[event.rectangle[3]]
        if y0 == y1:
            continue
        offset = 1 if event.is_start else -1
        tracker.modify_interval(
            y0, y1, offset, event.x, event.rect_idx, event.is_from_list_a
        )

    tracker.find_overlaps_between_lists()
    return _extract_overlap_results(tracker.overlaps)


def _extract_overlap_results(overlaps):
    indices_a = set()
    indices_b = set()
    pairs = set()
    for _, _, _, _, rects_a, rects_b in overlaps:
        indices_a.update(rects_a)
        indices_b.update(rects_b)
        for a_idx in rects_a:
            for b_idx in rects_b:
                pairs.add((a_idx, b_idx))
    if _is_empty(indices_a):
        return None, None, []
    return sorted(indices_a), sorted(indices_b), sorted(pairs)


def is_contained(rect_b, rect_a) -> bool:
    left_ok = rect_b[0] >= rect_a[0]
    right_ok = rect_b[2] <= rect_a[2]
    top_ok = rect_b[1] >= rect_a[1]
    bottom_ok = rect_b[3] <= rect_a[3]
    return left_ok and right_ok and top_ok and bottom_ok


def _has_overlap(rect_b, rect_a) -> bool:
    h_overlap = rect_b[0] < rect_a[2] and rect_b[2] > rect_a[0]
    v_overlap = rect_b[1] < rect_a[3] and rect_b[3] > rect_a[1]
    return h_overlap and v_overlap


def classify_intersection(rect_b, rect_a) -> dict[str, Any]:
    if is_contained(rect_b, rect_a):
        return _make_intersection_result("contained", [], [], [])
    if not _has_overlap(rect_b, rect_a):
        return _make_intersection_result("none", [], [], [])

    crosses = {
        "left": rect_b[0] < rect_a[0] and rect_b[2] > rect_a[0],
        "right": rect_b[0] < rect_a[2] and rect_b[2] > rect_a[2],
        "top": rect_b[1] < rect_a[1] and rect_b[3] > rect_a[1],
        "bottom": rect_b[1] < rect_a[3] and rect_b[3] > rect_a[3],
    }
    vertical = [name for name in ("left", "right") if crosses[name]]
    horizontal = [name for name in ("top", "bottom") if crosses[name]]
    boundaries = vertical + horizontal
    itype = _determine_intersection_type(bool(vertical), bool(horizontal))
    return _make_intersection_result(itype, boundaries, horizontal, vertical)


def _determine_intersection_type(crosses_vertical, crosses_horizontal) -> str:
    if crosses_vertical and crosses_horizontal:
        return "both"
    if crosses_vertical:
        return "vertical"
    if crosses_horizontal:
        return "horizontal"
    return "none"


def _make_intersection_result(itype, boundaries, horizontal, vertical):
    return {
        "type": itype,
        "boundaries_crossed": boundaries,
        "horizontal_boundaries": horizontal,
        "vertical_boundaries": vertical,
    }


def analyze_b_intersections(rectangles_a, rectangles_b, b_idx_to_ann_id=None):
    _, _, overlap_pairs = find_overlapping_between_lists(rectangles_a, rectangles_b)
    if _is_empty(overlap_pairs):
        return {}

    normalized_a = [_normalize_rect(r) for r in rectangles_a]
    normalized_b = [_normalize_rect(r) for r in rectangles_b]

    grouped_by_b = defaultdict(list)
    for a_idx, b_idx in overlap_pairs:
        grouped_by_b[b_idx].append(a_idx)

    result = {}
    for b_idx, a_indices in grouped_by_b.items():
        intersections = _get_intersections_for_b(
            normalized_b[b_idx], a_indices, normalized_a
        )
        _add_to_result_if_valid(result, b_idx, intersections, b_idx_to_ann_id)
    return result


def _get_intersections_for_b(rect_b, a_indices, normalized_a):
    intersections = []
    excluded_types = {"contained", "none"}
    for a_idx in a_indices:
        info = classify_intersection(rect_b, normalized_a[a_idx])
        if info["type"] in excluded_types:
            continue
        intersections.append(
            {
                "rect_a_index": a_idx,
                "intersection_type": info["type"],
                "boundaries_crossed": info["boundaries_crossed"],
                "horizontal_boundaries": info["horizontal_boundaries"],
                "vertical_boundaries": info["vertical_boundaries"],
            }
        )
    return intersections


def _add_to_result_if_valid(result, b_idx, intersections, b_idx_to_ann_id):
    if _is_empty(intersections):
        return
    b_key = b_idx_to_ann_id[b_idx] if b_idx_to_ann_id else b_idx
    result[b_key] = {
        "intersecting_rectangles_a": intersections,
        "intersection_count": len(intersections),
    }


def load_coco_data_for_processing(coco_api: COCO, img_ids: list[int]):
    rectangles_b = []
    b_idx_to_ann_id = {}
    current_idx = 0
    for img_id in img_ids:
        ann_ids = coco_api.getAnnIds(imgIds=img_id)
        for ann in coco_api.loadAnns(ann_ids):
            bbox = ann["bbox"]
            rectangles_b.append(
                [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            )
            b_idx_to_ann_id[current_idx] = ann["id"]
            current_idx += 1
    return rectangles_b, b_idx_to_ann_id


def validate_coco_ids(coco_data: dict[str, Any]) -> dict[str, Any]:
    image_ids = [img["id"] for img in coco_data["images"]]
    ann_ids = [ann["id"] for ann in coco_data["annotations"]]
    cat_ids = [cat["id"] for cat in coco_data["categories"]]
    return {
        "images_unique": len(image_ids) == len(set(image_ids)),
        "annotations_unique": len(ann_ids) == len(set(ann_ids)),
        "categories_unique": len(cat_ids) == len(set(cat_ids)),
        "num_images": len(image_ids),
        "num_annotations": len(ann_ids),
        "num_categories": len(cat_ids),
    }


def get_id_summary(coco_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "images": _summarize_ids([img["id"] for img in coco_data["images"]]),
        "annotations": _summarize_ids([a["id"] for a in coco_data["annotations"]]),
        "categories": _summarize_ids([c["id"] for c in coco_data["categories"]]),
    }


def _summarize_ids(id_list: list[int]) -> dict[str, Any]:
    if _is_empty(id_list):
        return {"count": 0, "min_id": None, "max_id": None}
    return {"count": len(id_list), "min_id": min(id_list), "max_id": max(id_list)}


def find_duplicate_ids(coco_data: dict[str, Any]) -> dict[str, list[int]]:
    return {
        "duplicate_image_ids": _find_duplicates(
            [img["id"] for img in coco_data["images"]]
        ),
        "duplicate_annotation_ids": _find_duplicates(
            [ann["id"] for ann in coco_data["annotations"]]
        ),
        "duplicate_category_ids": _find_duplicates(
            [cat["id"] for cat in coco_data["categories"]]
        ),
    }


def _find_duplicates(id_list: list[int]) -> list[int]:
    seen = set()
    duplicates = set()
    for item in id_list:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)


def get_category_distribution(coco_data: dict[str, Any]) -> dict[int, int]:
    distribution: dict[int, int] = defaultdict(int)
    for ann in coco_data["annotations"]:
        distribution[ann["category_id"]] += 1
    return dict(distribution)


def get_annotations_per_image(coco_data: dict[str, Any]) -> dict[int, int]:
    distribution: dict[int, int] = defaultdict(int)
    for ann in coco_data["annotations"]:
        distribution[ann["image_id"]] += 1
    return dict(distribution)


def filter_annotations_by_category(
    coco_data: dict[str, Any], category_ids: list[int]
) -> list[dict[str, Any]]:
    category_set = set(category_ids)
    return [a for a in coco_data["annotations"] if a["category_id"] in category_set]


def filter_annotations_by_image(
    coco_data: dict[str, Any], image_ids: list[int]
) -> list[dict[str, Any]]:
    image_set = set(image_ids)
    return [a for a in coco_data["annotations"] if a["image_id"] in image_set]


def get_category_name_map(coco_data: dict[str, Any]) -> dict[int, str]:
    return {cat["id"]: cat["name"] for cat in coco_data["categories"]}


def get_image_filename_map(coco_data: dict[str, Any]) -> dict[int, str]:
    return {img["id"]: img["file_name"] for img in coco_data["images"]}


def get_images_without_annotations(coco_data: dict[str, Any]) -> list[int]:
    annotated = {ann["image_id"] for ann in coco_data["annotations"]}
    all_images = {img["id"] for img in coco_data["images"]}
    return list(all_images - annotated)


def get_annotation_area_stats(coco_data: dict[str, Any]) -> dict[str, Any]:
    areas = [ann["area"] for ann in coco_data["annotations"]]
    if _is_empty(areas):
        return {"count": 0, "min_area": None, "max_area": None, "mean_area": None}
    return {
        "count": len(areas),
        "min_area": min(areas),
        "max_area": max(areas),
        "mean_area": sum(areas) / len(areas),
    }


class CocoDatasetGenerator:

    def __init__(
        self,
        config: dict,
        bloom_settings: BloomSettings = BloomSettings(),
        use_bloom_filter: bool = True,
    ):
        self.config = config
        self.bloom_settings = bloom_settings
        self.coco_data = self._initialize_coco_structure()
        self.annotation_id = 1
        self.image_id = 1
        self.annotation_bloom = self._build_bloom_filter(use_bloom_filter)
        os.makedirs(self.config["image"]["output_dir"], exist_ok=True)

    def _build_bloom_filter(
        self, use_bloom_filter: bool
    ) -> Optional[AnnotationBloomFilter]:
        if not use_bloom_filter:
            return None
        estimated = self._estimate_total_annotations()
        capacity = int(estimated * self.bloom_settings.capacity_multiplier)
        return AnnotationBloomFilter(capacity=capacity, settings=self.bloom_settings)

    def _estimate_total_annotations(self) -> int:
        img_cfg = self.config["image"]
        poly_cfg = self.config["polygon"]
        avg_polygons = (
            poly_cfg["min_num_polygons"] + poly_cfg["max_num_polygons"]
        ) / 2
        return int(img_cfg["num_images"] * avg_polygons)

    def _initialize_coco_structure(self) -> dict:
        cats_cfg = self.config["categories"]
        categories = self._build_categories(
            cats_cfg["num_supercategories"],
            cats_cfg["category_prefix"],
            cats_cfg["supercategory_prefix"],
        )
        return {
            "info": self.config.get("metadata", {}),
            "licenses": [self.config.get("metadata", {}).get("license", {})],
            "categories": categories,
            "images": [],
            "annotations": [],
        }

    def _build_categories(
        self, num_supercats: int, cat_prefix: str, supercat_prefix: str
    ) -> list[dict]:
        return [
            {
                "id": i + 1,
                "name": f"{cat_prefix}_{i + 1}",
                "supercategory": f"{supercat_prefix}_{(i % num_supercats) + 1}",
            }
            for i in range(self.config["categories"]["num_categories"])
        ]

    def run(self) -> tuple[str, str]:
        img_cfg = self.config["image"]
        for i in range(img_cfg["num_images"]):
            self._process_image()
            logger.info("Generated image %d/%d", i + 1, img_cfg["num_images"])
        self._save_json()
        self._log_bloom_stats_if_enabled()
        logger.info("Dataset generation complete.")
        return self.config["output"]["coco_json_path"], img_cfg["output_dir"]

    def _log_bloom_stats_if_enabled(self) -> None:
        if self.annotation_bloom is None:
            return
        stats = self.annotation_bloom.get_stats()
        logger.info(
            "Bloom stats: %d tracked, %.4f MB, %.1f%% utilization",
            stats["items_added"],
            stats["memory_mb"],
            stats["utilization"] * 100,
        )

    def _process_image(self) -> None:
        img_cfg = self.config["image"]
        width = self._calculate_dimension(img_cfg["width"], img_cfg["width_variation"])
        height = self._calculate_dimension(
            img_cfg["height"], img_cfg["height_variation"]
        )
        image = Image.new("RGB", (width, height), tuple(img_cfg["background_color"]))
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        file_name = (
            f"image_{self.image_id:05d}.{self.config['output']['image_format']}"
        )
        self._add_image_entry(file_name, width, height)
        self._add_polygons_to_image(draw, width, height)

        image.paste(overlay, (0, 0), overlay)
        self._save_image(image, file_name)
        self.image_id += 1

    def _calculate_dimension(self, base: int, variation: int) -> int:
        return base + random.randint(-variation, variation)

    def _add_polygons_to_image(
        self, draw: ImageDraw.ImageDraw, width: int, height: int
    ) -> None:
        poly_cfg = self.config["polygon"]
        num_polygons = random.randint(
            poly_cfg["min_num_polygons"], poly_cfg["max_num_polygons"]
        )
        for _ in range(num_polygons):
            self._add_polygon(draw, width, height)

    def _add_polygon(
        self, draw: ImageDraw.ImageDraw, width: int, height: int
    ) -> None:
        poly_cfg = self.config["polygon"]
        color_cfg = self.config["colors"]
        min_dim = min(width, height)
        min_r = min_dim * poly_cfg["min_radius_ratio"]
        max_r = min_dim * poly_cfg["max_radius_ratio"]
        margin = max_r * 1.1
        center = (
            np.random.uniform(margin, width - margin),
            np.random.uniform(margin, height - margin),
        )

        gen = PolygonGenerator(
            num_vertices=random.randint(
                poly_cfg["min_vertices"], poly_cfg["max_vertices"]
            ),
            min_radius=min_r,
            max_radius=max_r,
            center=center,
            angle_variation=poly_cfg["angle_variation"],
        )
        points = gen.generate_polygon()
        flat_points = points.flatten().tolist()

        draw.polygon(
            flat_points,
            fill=tuple(random.choice(color_cfg["polygons"])),
            outline=tuple(color_cfg["outline"]),
            width=color_cfg["outline_width"],
        )
        self._add_annotation_entry(points, flat_points)

    def _add_image_entry(self, file_name: str, width: int, height: int) -> None:
        license_id = self.config["metadata"].get("license", {}).get("id", 1)
        self.coco_data["images"].append(
            {
                "id": self.image_id,
                "width": width,
                "height": height,
                "file_name": file_name,
                "license": license_id,
                "date_captured": str(datetime.datetime.now()),
            }
        )

    def _add_annotation_entry(
        self, points: np.ndarray, segmentation: list[float]
    ) -> None:
        self.coco_data["annotations"].append(
            {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": random.choice(self.coco_data["categories"])["id"],
                "segmentation": [segmentation],
                "area": PolygonGenerator.area(points),
                "bbox": PolygonGenerator.bounding_box(points),
                "iscrowd": 0,
            }
        )
        self._register_annotation_id(self.annotation_id)
        self.annotation_id += 1

    def _register_annotation_id(self, ann_id: int) -> None:
        if self.annotation_bloom is not None:
            self.annotation_bloom.add(ann_id)

    def check_annotation_id_collision(self, ann_id: int) -> bool:
        if self.annotation_bloom is not None:
            return self.annotation_bloom.might_exist(ann_id)
        return False

    def _save_image(self, image: Image.Image, file_name: str) -> None:
        path = os.path.join(self.config["image"]["output_dir"], file_name)
        image.save(path, quality=self.config["output"]["image_quality"])

    def _save_json(self) -> None:
        with open(self.config["output"]["coco_json_path"], "w") as f:
            json.dump(self.coco_data, f, indent=2)

    def get_bloom_filter(self) -> Optional[AnnotationBloomFilter]:
        return self.annotation_bloom

    def export_bloom_state(self) -> Optional[dict[str, Any]]:
        if self.annotation_bloom is not None:
            return self.annotation_bloom.export_state()
        return None


class COCODatasetExtender:

    def __init__(
        self,
        existing_coco_path: str,
        bloom_settings: BloomSettings = BloomSettings(),
    ):
        self.coco_path = existing_coco_path
        self.bloom_settings = bloom_settings
        self.coco_data = self._load_coco_data(existing_coco_path)
        self._extract_existing_ids()
        self._calculate_max_ids()
        self._initialize_bloom_filter()

    def _load_coco_data(self, path: str) -> dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    def _extract_existing_ids(self) -> None:
        self.existing_image_ids = {img["id"] for img in self.coco_data["images"]}
        self.existing_annotation_ids = {
            ann["id"] for ann in self.coco_data["annotations"]
        }
        self.existing_category_ids = {
            cat["id"] for cat in self.coco_data["categories"]
        }

    def _calculate_max_ids(self) -> None:
        self.max_image_id = self._get_max_id(self.existing_image_ids)
        self.max_annotation_id = self._get_max_id(self.existing_annotation_ids)
        self.max_category_id = self._get_max_id(self.existing_category_ids)

    def _get_max_id(self, id_set: set[int]) -> int:
        if _is_empty(id_set):
            return 0
        return max(id_set)

    def _initialize_bloom_filter(self) -> None:
        n_existing = len(self.existing_annotation_ids)
        expected_additions = int(n_existing * self.bloom_settings.expected_growth_ratio)
        capacity = int(
            (n_existing + expected_additions) * self.bloom_settings.capacity_multiplier
        )
        self.annotation_bloom = AnnotationBloomFilter(
            capacity=max(1, capacity), settings=self.bloom_settings
        )
        for ann_id in self.existing_annotation_ids:
            self.annotation_bloom.add(ann_id)

    def check_annotation_id(self, ann_id: int) -> tuple[bool, Optional[int]]:
        if self.annotation_bloom.definitely_new(ann_id):
            return True, None
        return self._handle_potential_collision(ann_id)

    def _handle_potential_collision(self, ann_id: int) -> tuple[bool, Optional[int]]:
        is_real_collision = ann_id in self.existing_annotation_ids
        if is_real_collision:
            self.max_annotation_id += 1
            return False, self.max_annotation_id
        return True, None

    def add_annotations(
        self,
        new_annotations: list[dict[str, Any]],
        auto_remap_ids: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        stats = self._create_annotation_stats(len(new_annotations))
        processed = [
            self._process_single_annotation(ann, auto_remap_ids, stats)
            for ann in new_annotations
        ]
        return processed, stats

    def _create_annotation_stats(self, total: int) -> dict[str, Any]:
        return {
            "total": total,
            "collisions_detected": 0,
            "false_positives": 0,
            "ids_remapped": 0,
            "remapping": {},
        }

    def _process_single_annotation(
        self, ann: dict[str, Any], auto_remap_ids: bool, stats: dict[str, Any]
    ) -> dict[str, Any]:
        original_id = ann["id"]
        is_unique, suggested_id = self.check_annotation_id(original_id)
        if is_unique:
            return self._add_unique_annotation(ann, original_id)
        return self._handle_collision(
            ann, original_id, suggested_id, auto_remap_ids, stats
        )

    def _add_unique_annotation(
        self, ann: dict[str, Any], original_id: int
    ) -> dict[str, Any]:
        self.annotation_bloom.add(original_id)
        self.existing_annotation_ids.add(original_id)
        return ann

    def _handle_collision(
        self,
        ann: dict[str, Any],
        original_id: int,
        suggested_id: Optional[int],
        auto_remap_ids: bool,
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        stats["collisions_detected"] += 1
        can_remap = auto_remap_ids and suggested_id is not None
        if can_remap:
            return self._remap_annotation(ann, original_id, suggested_id, stats)
        raise ValueError(f"Annotation ID collision: {original_id} already exists")

    def _remap_annotation(
        self,
        ann: dict[str, Any],
        original_id: int,
        suggested_id: int,
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        new_ann = ann.copy()
        new_ann["id"] = suggested_id
        self.annotation_bloom.add(suggested_id)
        self.existing_annotation_ids.add(suggested_id)
        stats["ids_remapped"] += 1
        stats["remapping"][original_id] = suggested_id
        warnings.warn(f"Annotation ID collision: {original_id} -> {suggested_id}")
        return new_ann

    def merge_and_save(
        self,
        new_coco_path: str,
        output_path: str,
        auto_remap_ids: bool = True,
    ) -> dict[str, Any]:
        new_coco = self._load_coco_data(new_coco_path)
        processed_annotations, ann_stats = self.add_annotations(
            new_coco["annotations"], auto_remap_ids=auto_remap_ids
        )

        merged_data = self.coco_data.copy()
        merged_data["annotations"].extend(processed_annotations)

        image_id_map = self._merge_images(new_coco["images"], merged_data)
        self._update_refs(processed_annotations, image_id_map, "image_id")

        category_id_map = self._merge_categories(new_coco["categories"], merged_data)
        self._update_refs(processed_annotations, category_id_map, "category_id")

        self._save_merged_data(merged_data, output_path)
        return self._compile_merge_stats(
            ann_stats, new_coco, image_id_map, category_id_map, output_path
        )

    def _merge_images(
        self, new_images: list[dict[str, Any]], merged_data: dict[str, Any]
    ) -> dict[int, int]:
        image_id_map = {}
        for img in new_images:
            is_collision = img["id"] in self.existing_image_ids
            if is_collision:
                self.max_image_id += 1
                image_id_map[img["id"]] = self.max_image_id
                img["id"] = self.max_image_id
            self.existing_image_ids.add(img["id"])
            merged_data["images"].append(img)
        return image_id_map

    def _update_refs(
        self,
        annotations: list[dict[str, Any]],
        id_map: dict[int, int],
        ref_key: str,
    ) -> None:
        if _is_empty(id_map):
            return
        for ann in annotations:
            if ann[ref_key] in id_map:
                ann[ref_key] = id_map[ann[ref_key]]

    def _merge_categories(
        self, new_categories: list[dict[str, Any]], merged_data: dict[str, Any]
    ) -> dict[int, int]:
        category_name_map = {
            cat["name"]: cat["id"] for cat in merged_data["categories"]
        }
        category_id_map = {}
        for cat in new_categories:
            if cat["name"] in category_name_map:
                category_id_map[cat["id"]] = category_name_map[cat["name"]]
                continue
            self.max_category_id += 1
            category_id_map[cat["id"]] = self.max_category_id
            cat["id"] = self.max_category_id
            merged_data["categories"].append(cat)
            category_name_map[cat["name"]] = self.max_category_id
        return category_id_map

    def _save_merged_data(
        self, merged_data: dict[str, Any], output_path: str
    ) -> None:
        with open(output_path, "w") as f:
            json.dump(merged_data, f, indent=2)

    def _compile_merge_stats(
        self,
        ann_stats: dict[str, Any],
        new_coco: dict[str, Any],
        image_id_map: dict[int, int],
        category_id_map: dict[int, int],
        output_path: str,
    ) -> dict[str, Any]:
        return {
            "annotations": ann_stats,
            "images_added": len(new_coco["images"]),
            "images_remapped": len(image_id_map),
            "categories_added": self._count_new_categories(new_coco["categories"]),
            "bloom_filter_stats": self.annotation_bloom.get_stats(),
            "output_path": output_path,
        }

    def _count_new_categories(self, categories: list[dict[str, Any]]) -> int:
        existing_names = {cat["name"] for cat in self.coco_data["categories"]}
        return sum(1 for cat in categories if cat["name"] not in existing_names)


class CocoResolutionSplitter:

    def __init__(self, settings: SplitterSettings):
        self.settings = settings
        self.input_path = Path(settings.input_json_path)
        self.output_dir = Path(settings.output_dir)
        self._validate_input_path()

        logger.info("Loading annotation file: %s", self.input_path)
        self.coco = COCO(str(self.input_path))
        self.base_info = self.coco.dataset.get("info", {})
        self.base_licenses = self.coco.dataset.get("licenses", [])
        self.base_categories = self.coco.dataset.get("categories", [])
        self.name_part = self.input_path.stem
        self.ext_part = self.input_path.suffix

    def _validate_input_path(self) -> None:
        if self.input_path.exists():
            return
        raise FileNotFoundError(f"Input file not found: {self.input_path}")

    def _group_images_by_resolution(self) -> defaultdict[tuple[int, int], list[int]]:
        logger.info("Grouping images by resolution...")
        res_to_img_ids: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        for img_id in self.coco.getImgIds():
            img = self.coco.loadImgs([img_id])[0]
            res_to_img_ids[(img["width"], img["height"])].append(img_id)
        logger.info("Found %d unique resolutions.", len(res_to_img_ids))
        return res_to_img_ids

    def _build_output_path(self, width: int, height: int) -> Path:
        filename = (
            f"{self.name_part}{self.settings.filename_separator}"
            f"{width}{self.settings.resolution_separator}{height}{self.ext_part}"
        )
        return self.output_dir / filename

    def _create_subset(self, img_ids: list[int]) -> dict[str, Any]:
        ann_ids = self.coco.getAnnIds(imgIds=img_ids)
        return {
            "info": self.base_info,
            "licenses": self.base_licenses,
            "categories": self.base_categories,
            "images": self.coco.loadImgs(img_ids),
            "annotations": self.coco.loadAnns(ann_ids),
        }

    def _save_subset(self, subset_data: dict[str, Any], output_path: Path) -> None:
        logger.info("Saving to %s", output_path)
        with open(output_path, "w") as f:
            json.dump(subset_data, f, indent=self.settings.json_indent)

    def run(self) -> dict[tuple[int, int], Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        resolutions_map = self._group_images_by_resolution()
        resolution_file_map = {
            res: self._build_output_path(*res) for res in resolutions_map
        }
        for resolution, img_ids in resolutions_map.items():
            logger.info(
                "Processing resolution %dx%d (%d images)",
                resolution[0],
                resolution[1],
                len(img_ids),
            )
            self._save_subset(
                self._create_subset(img_ids), resolution_file_map[resolution]
            )
        logger.info("Splitting complete.")
        return resolution_file_map

    @classmethod
    def from_config(cls, config_path: Union[Path, str]) -> "CocoResolutionSplitter":
        config = load_yaml(Path(config_path))
        settings_dict = config.get("coco_splitter_settings")
        if settings_dict is None:
            raise KeyError(f"Key 'coco_splitter_settings' not found in {config_path}.")
        return cls(hydrate(SplitterSettings, settings_dict))


class PreProcess:

    def __init__(self, settings: SliceSettings):
        self.settings = settings
        self.unique_resolutions = self._get_unique_resolutions(
            settings.ann_file_path
        )
        self.resolution_to_overlap = self._get_overlap_ratios_per_resolution()
        self.resolution_to_slices = generate_overlapping_slices_sweepline(
            self.resolution_to_overlap
        )

    def _calculate_overlap_ratios(
        self, img_size: tuple[int, int]
    ) -> tuple[float, float]:
        x_res, y_res = img_size
        sw, sh = self.settings.tile_width, self.settings.tile_height
        x_mod, y_mod = x_res % sw, y_res % sh
        return (
            x_mod / sw if x_mod else 0.0,
            y_mod / sh if y_mod else 0.0,
        )

    def _get_unique_resolutions(self, ann_file_path: str) -> set[tuple[int, int]]:
        coco = COCO(ann_file_path)
        return {(img["width"], img["height"]) for img in coco.imgs.values()}

    def _get_overlap_ratios_per_resolution(
        self,
    ) -> dict[tuple[int, int], tuple[float, float]]:
        return {
            resolution: self._calculate_overlap_ratios(resolution)
            for resolution in self.unique_resolutions
        }

    @classmethod
    def from_config(cls, config_path: Union[Path, str]) -> "PreProcess":
        config = load_yaml(Path(config_path))
        settings_dict = config.get("preprocess_settings")
        if settings_dict is None:
            raise KeyError(f"Key 'preprocess_settings' not found in {config_path}.")
        return cls(hydrate(SliceSettings, settings_dict))


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="COCO dataset slicing and validation toolkit.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("config_path", type=Path, help="Path to config.yaml")
    parser.add_argument(
        "command",
        choices=["split", "preprocess"],
        help="Pipeline stage to run.",
    )
    args = parser.parse_args()

    try:
        if args.command == "split":
            CocoResolutionSplitter.from_config(args.config_path).run()
            return
        pre = PreProcess.from_config(args.config_path)
        logger.info(
            "Generated slices for %d resolutions.", len(pre.resolution_to_slices)
        )
    except Exception as e:
        logger.exception("An error occurred: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()