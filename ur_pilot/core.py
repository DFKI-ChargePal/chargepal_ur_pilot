from __future__ import annotations

# global
from time import process_time
import numpy as np
from pathlib import Path
from enum import auto, Enum
from contextlib import contextmanager

import chargepal_aruco as ca
from rigmopy import Pose, Vector6d

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

    def plug_in(self, wrench: Vector6d, compliant_axes: list[int], time_out: float) -> tuple[bool, Pose]:
        self._check_control_context(expected=[ControlContext.FORCE, ControlContext.MOTION])

        if self.control_context is ControlContext.FORCE:
            # Wrench will be applied with respect to the current TCP pose
            X_tcp = self.robot.get_tcp_pose()
            task_frame = X_tcp.xyz + X_tcp.axis_angle
            x_ref = np.array(X_tcp.xyz, dtype=np.float32)
            # Time observation
            fin = False
            t_ref = t_start = process_time()
            while True:
                # Apply wrench
                self.robot.force_mode(
                    task_frame=task_frame,
                    selection_vector=compliant_axes,
                    wrench=wrench.xyzXYZ)
                t_now = process_time()
                # Check every second if robot is still moving
                if t_now - t_ref > 1.0:
                    x_now = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
                    if np.allclose(x_ref, x_now, atol=0.001):
                        fin = True
                        break
                    t_ref, x_ref = t_now, x_now
                if t_now - t_start > time_out:
                    break
                if ca.EventObserver.state is ca.EventObserver.Type.QUIT:
                    break
            # Stop robot movement.
            self.robot.force_mode(task_frame=task_frame, selection_vector=6*[0], wrench=6*[0.0])
            return fin, self.robot.get_tcp_pose()
        elif self.control_context is ControlContext.MOTION:
            raise NotImplementedError(f"Control context hasn't been implemented yet for this action.")
        else:
            raise RuntimeError(f"Undefined program state.")

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
