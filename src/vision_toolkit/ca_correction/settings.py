from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"

GREEN, RED, BLUE = 0, 1, 2
CHANNEL_COUNT = 3
MODE_STANDARD, MODE_BRIGHTEN_ONLY, MODE_DARKEN_ONLY = 0, 1, 2

GUIDE_MAP = {"green": GREEN, "red": RED, "blue": BLUE}
MODE_MAP = {
    "standard": MODE_STANDARD,
    "brighten": MODE_BRIGHTEN_ONLY,
    "darken": MODE_DARKEN_ONLY,
}


@dataclass(frozen=True)
class CADefaults:
    radius: int = 5
    strength: float = 1.0
    safety: float = 0.5
    log_threshold: float = 2.0


@dataclass(frozen=True)
class NumericalSettings:
    epsilon_small: float = 1e-6
    epsilon_tiny: float = 1e-8
    epsilon_filter: float = 1e-4
    clip_min: float = 0.0
    clip_max_multiplier: float = 2.0
    normalize_threshold: float = 1.0


@dataclass(frozen=True)
class SafetyBlenderSettings:
    threshold: float = 0.5
    blend_min: float = 0.0
    blend_max: float = 1.0


@dataclass(frozen=True)
class KernelSettings:
    size_multiplier: int = 2
    size_offset: int = 1


@dataclass(frozen=True)
class CASettings:
    defaults: CADefaults = field(default_factory=CADefaults)
    numerical: NumericalSettings = field(default_factory=NumericalSettings)
    safety_blender: SafetyBlenderSettings = field(
        default_factory=SafetyBlenderSettings
    )
    kernel: KernelSettings = field(default_factory=KernelSettings)


def _hydrate(cls: type, mapping: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in mapping.items() if k in known})


def load_ca_settings(path: Path = CONFIG_PATH) -> CASettings:
    if not path.exists():
        return CASettings()
    raw = yaml.safe_load(path.read_text()) or {}
    return CASettings(
        defaults=_hydrate(CADefaults, raw.get("defaults", {})),
        numerical=_hydrate(NumericalSettings, raw.get("numerical", {})),
        safety_blender=_hydrate(
            SafetyBlenderSettings, raw.get("safety_blender", {})
        ),
        kernel=_hydrate(KernelSettings, raw.get("kernel", {})),
    )