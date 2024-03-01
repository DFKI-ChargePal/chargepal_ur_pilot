from __future__ import annotations

# global
from dataclasses import dataclass

# typing
from typing import Sequence


@dataclass
class TargetConfig:
    xyz: Sequence[float]
    xyzw: Sequence[float]
