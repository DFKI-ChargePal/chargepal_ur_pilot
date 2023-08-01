from __future__ import annotations

# global
import numpy as np
from pathlib import Path
from enum import auto, Enum
from contextlib import contextmanager
from rigmopy import Pose

# local
from ur_pilot.robot import Robot

# typing
from typing import Iterator, Sequence



@contextmanager
def connect(config: Path | None = None) -> Iterator[Pilot]:
    pilot = Pilot(config)
    try:
        yield pilot
    finally:
        pilot.exit()


class ControlContext(Enum):
    DISABLED = auto()
    FORCE = auto()
    MOTION = auto()
    POSITION = auto()
    VELOCITY = auto()
    TEACH_IN = auto()


class Pilot:

    def __init__(self, config: Path | None = None) -> None:
        """ Core class to interact with the robot

        Args:
            config: Path to a configuration toml file

        Raises:
            FileNotFoundError: Check if configuration file exists
        """
        if config is not None and not config.exists():
            raise FileNotFoundError(f"Configuration with file path {config} not found.")
        self.robot = Robot(config)
        self.control_context = ControlContext.DISABLED

    def _check_control_context(self, expected: ControlContext | list[ControlContext]) -> None:
        if self.control_context is ControlContext.DISABLED:
            raise RuntimeError(f"Pilot is not in any control context. Running actions is not possible.")
        elif type(expected) == list and self.control_context not in expected:
            raise RuntimeError(f"This action is not able to use one of the control context '{self.control_context}'")
        elif self.control_context is not expected:
            raise RuntimeError(f"This action is not able to use the control context '{self.control_context}'")

    @contextmanager
    def position_control(self) -> Iterator[None]:
        self.control_context = ControlContext.POSITION
        yield
        self.control_context = ControlContext.DISABLED

    @contextmanager
    def force_control(self) -> Iterator[None]:
        self.robot.set_up_force_mode()
        self.control_context = ControlContext.FORCE
        yield
        self.robot.stop_force_mode()
        self.control_context = ControlContext.DISABLED

    @contextmanager
    def teach_in_control(self) -> Iterator[None]:
        self.robot.set_up_teach_mode()
        self.control_context = ControlContext.TEACH_IN
        yield
        self.robot.stop_teach_mode()
        self.control_context = ControlContext.DISABLED

    def move_home(self) -> list[float]:
        self._check_control_context(expected=ControlContext.POSITION)
        self.robot.move_home()
        new_j_pos = self.robot.get_joint_pos()
        return new_j_pos

    def move_to_joint_pos(self, q: Sequence[float]) -> list[float]:
        self._check_control_context(expected=ControlContext.POSITION)
        # Move to requested joint position
        self.robot.move_j(q)
        new_joint_pos = self.robot.get_joint_pos()
        return new_joint_pos

    def move_to_tcp_pose(self, target: Pose) -> Pose:
        self._check_control_context(expected=ControlContext.POSITION)
        # Move to requested TCP pose
        self.robot.move_l(target)
        new_pose = self.robot.get_tcp_pose()
        return new_pose

    def move_joints_random(self) -> list[float]:
        self._check_control_context(expected=ControlContext.POSITION)
        # Move to home joint position
        self.robot.move_home()
        # Move to random joint positions near to the home configuration
        home_q = np.array(self.robot.home_joint_config, dtype=np.float32)
        tgt_joint_q: list[float] = (home_q + (np.random.rand(6) * 2.0 - 1.0) * 0.075).tolist()
        self.robot.move_j(tgt_joint_q)
        # Move back to home joint positions
        self.robot.move_home()
        return tgt_joint_q

    def exit(self) -> None:
        """ Exit function which will be called from the context manager at the end """
        self.robot.exit()
