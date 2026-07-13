from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np

from vision_toolkit.coco_tools import SegmentTreeBase, _normalize_rect


@dataclass(frozen=True)
class RectSweepEvent:
    x: float
    rectangle: tuple
    is_start: bool
    rect_idx: int

    @property
    def sort_key(self):
        return (self.x, not self.is_start, self.rectangle[1])


class CoverageDepthTracker(SegmentTreeBase):

    def __init__(self, interval_lengths: list, y_coords: list):
        super().__init__(interval_lengths, y_coords)
        self.overlaps = []
        self.sweep_events = []
        self.active_intervals: dict[int, int] = {}
        self.active_rectangles: dict[int, set] = {}

    def modify_interval(self, i, k, offset, x_coord, rect_idx):
        for y in range(i, k):
            self.active_intervals.setdefault(y, 0)
            active = self.active_rectangles.setdefault(y, set())

            old_count = self.active_intervals[y]
            self.active_intervals[y] += offset
            new_count = self.active_intervals[y]

            if offset == 1:
                active.add(rect_idx)
            else:
                active.discard(rect_idx)

            if old_count != new_count:
                self.sweep_events.append(
                    (x_coord, y, old_count, new_count, set(active))
                )
        self._change(1, 0, self.N, i, k, offset)

    def find_overlaps(self):
        self.sweep_events.sort()
        active_regions: dict[int, dict] = {}

        for x, y, old_count, new_count, rect_set in self.sweep_events:
            if new_count < old_count:
                for count in range(old_count, new_count, -1):
                    self._close_region(active_regions, count, y, x)
            if new_count > old_count:
                for count in range(old_count + 1, new_count + 1):
                    if count >= 2:
                        active_regions.setdefault(count, {})[y] = (x, rect_set)

        self.overlaps = sorted(
            [o for o in self.overlaps if o[1] > o[0]],
            key=lambda o: (-o[4], o[0], o[2]),
        )

    def _close_region(self, active_regions, count, y, x):
        if count < 2 or y not in active_regions.get(count, {}):
            return
        start_x, start_rects = active_regions[count][y]
        if x > start_x:
            y_low = self.y_coords[y]
            has_next = y + 1 < len(self.y_coords)
            y_high = self.y_coords[y + 1] if has_next else y_low
            self.overlaps.append((start_x, x, y_low, y_high, count, start_rects))
        del active_regions[count][y]


def _find_transitivity(overlapping_indices: list) -> list:
    g = nx.Graph()
    g.add_edges_from(
        edge
        for node_set in overlapping_indices
        for edge in itertools.combinations(node_set, 2)
    )
    return [tuple(comp) for comp in nx.connected_components(g)]


def _build_events(rectangles: list) -> list[RectSweepEvent]:
    events = [
        RectSweepEvent(rect[i], rect, is_start, idx)
        for idx, rect in enumerate(rectangles)
        for i, is_start in ((0, True), (2, False))
    ]
    return sorted(events, key=lambda e: e.sort_key)


def _check_y_containment(inner_y: tuple, outer_y: tuple) -> bool:
    return outer_y[0] <= inner_y[0] and inner_y[1] <= outer_y[1]


def find_overlapping(rectangles: list) -> tuple:
    if not rectangles:
        return None, []

    normalized = [_normalize_rect(r) for r in rectangles]
    events = _build_events(normalized)

    y_coords = sorted({y for rect in normalized for y in (rect[1], rect[3])})
    y_intervals = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]
    y_map = {val: idx for idx, val in enumerate(y_coords)}

    tracker = CoverageDepthTracker(y_intervals, y_coords)

    active_rects: dict[int, tuple[int, int]] = {}
    nesting_candidates: dict[int, set] = {}
    nested_pairs = []

    for event in events:
        y_bounds = (y_map[event.rectangle[1]], y_map[event.rectangle[3]])

        if event.is_start:
            potential_outers = {
                idx
                for idx, other_y in active_rects.items()
                if _check_y_containment(y_bounds, other_y)
            }
            if potential_outers:
                nesting_candidates[event.rect_idx] = potential_outers
            active_rects[event.rect_idx] = y_bounds
            tracker.modify_interval(*y_bounds, +1, event.x, event.rect_idx)
        else:
            if event.rect_idx in nesting_candidates:
                confirmed = [
                    (event.rect_idx, outer_idx)
                    for outer_idx in nesting_candidates[event.rect_idx]
                    if outer_idx in active_rects
                ]
                nested_pairs.extend(confirmed)
                del nesting_candidates[event.rect_idx]
            del active_rects[event.rect_idx]
            tracker.modify_interval(*y_bounds, -1, event.x, event.rect_idx)

    tracker.find_overlaps()

    overlapping_indices = [o[5] for o in tracker.overlaps]
    overlap_groups = (
        _find_transitivity(overlapping_indices) if overlapping_indices else None
    )
    nested_indices = sorted({pair[0] for pair in nested_pairs})
    return overlap_groups, nested_indices


def find_invalid_inds(areas: np.ndarray, overlap_groups: list) -> list:
    sorted_inds = np.argsort(areas)[::-1]

    def get_invalid_from_group(group):
        mask = np.isin(sorted_inds, list(group))
        max_idx = sorted_inds[np.argmax(mask)]
        return [idx for idx in group if idx != max_idx]

    invalid = itertools.chain.from_iterable(
        map(get_invalid_from_group, overlap_groups)
    )
    return list(set(invalid))


def find_all_invalid_inds(rectangles: list, areas: np.ndarray) -> tuple:
    overlap_groups, nested_indices = find_overlapping(rectangles)
    overlap_invalid = (
        find_invalid_inds(areas, overlap_groups) if overlap_groups else []
    )
    all_invalid = sorted(set(overlap_invalid) | set(nested_indices))
    return overlap_invalid, nested_indices, all_invalid