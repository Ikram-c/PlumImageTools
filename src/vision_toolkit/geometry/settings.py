from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass(frozen=True)
class GJK2DSettings:
    max_iterations: int = 128
    tolerance: float = 1e-6
    upper_bound: float = 1.79769e308
    use_nesterov_acceleration: bool = False


@dataclass(frozen=True)
class GJK3DSettings:
    max_iterations: int = 20
    epsilon: float = 1.19209290e-07


@dataclass(frozen=True)
class GeometrySettings:
    gjk_2d: GJK2DSettings = field(default_factory=GJK2DSettings)
    gjk_3d: GJK3DSettings = field(default_factory=GJK3DSettings)


def _hydrate(cls: type, mapping: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in mapping.items() if k in known})


def load_geometry_settings(path: Path = CONFIG_PATH) -> GeometrySettings:
    if not path.exists():
        return GeometrySettings()
    raw = yaml.safe_load(path.read_text()) or {}
    return GeometrySettings(
        gjk_2d=_hydrate(GJK2DSettings, raw.get("gjk_2d", {})),
        gjk_3d=_hydrate(GJK3DSettings, raw.get("gjk_3d", {})),
    )