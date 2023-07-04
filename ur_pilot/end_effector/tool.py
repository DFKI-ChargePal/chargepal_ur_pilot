from __future__ import annotations

# global
from rigmopy import Vector3d


class Tool:

    def __init__(self, mass: float, com: list[float]):
        self.mass = mass
        self.com = Vector3d().from_xyz(com)
        self.gravity = Vector3d().from_xyz([0.0, 0.0, -9.81])

    @property
    def f_inertia(self) -> Vector3d:
        return self.mass * self.gravity
