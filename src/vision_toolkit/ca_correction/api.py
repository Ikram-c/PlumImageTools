from __future__ import annotations

from numpy.typing import NDArray

from vision_toolkit.ca_correction.pipeline import CACorrectRGB
from vision_toolkit.ca_correction.settings import GUIDE_MAP, MODE_MAP


def correct_chromatic_aberration(
    image: NDArray,
    radius: int = None,
    strength: float = None,
    guide: str = "green",
    mode: str = "standard",
    refine: bool = True,
    safety: float = None,
) -> NDArray:
    corrector = CACorrectRGB(
        radius=radius,
        strength=strength,
        guide=GUIDE_MAP.get(guide.lower(), GUIDE_MAP["green"]),
        mode=MODE_MAP.get(mode.lower(), MODE_MAP["standard"]),
        refine_manifolds=refine,
        safety=safety,
    )
    return corrector(image)