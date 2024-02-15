from __future__ import annotations
# global
import time
import logging
import numpy as np
import spatialmath as sm
from pathlib import Path
from rigmopy import utils_math as rp_math
from rigmopy import Pose, Quaternion, Transformation, Vector3d, Vector6d

from ur_control.utils import clip
from ur_control.robots import RealURRobot
from ur_control.utils import tr2ur_format, ur_format2tr

# typing
from typing import Sequence
from numpy import typing as npt

LOGGER = logging.getLogger(__name__)


class Robot(RealURRobot):

    def __init__(self, cfg_path: Path) -> None:
        super().__init__(cfg_path)

    def movej_path(self,
                   wps: Sequence[npt.ArrayLike],
                   vel: float | None = None,
                   acc: float | None = None,
                   bf: float | None = None) -> None:
        """
        Move the robot in joint space along a specified path.

        Args:
            wps: Waypoints of the path. List of arrays of the target joint angles in radians
            vel: Joint speed of leading axis \(^{rad}/_{s^2}\). Defaults to None.
            acc: Joint acceleration of leading axis \(^{rad}/{s^2}\). Defaults to None.
            bf:  Blend factor to smooth movements.
        """
        wps_f32 = [np.array(np.reshape(target, 6), dtype=np.float32) for target in wps]
        speed = (
            clip(vel, 0.0, self.cfg.robot.joints.max_vel)
            if vel
            else self.cfg.robot.joints.vel
        )
        acceleration = (
            clip(acc, 0.0, self.cfg.robot.joints.max_acc)
            if acc
            else self.cfg.robot.joints.acc
        )
        bf = clip(bf, 0.0, 0.1) if bf else 0.02
        path = [[*tg.tolist(), speed, acceleration, bf] for tg in wps_f32]
        # Set blend factor of the last waypoint to zero to stop smoothly
        path[-1][-1] = 0.0
        success = self.rtde_controller.moveJ(path=path)

    ####################################
    #       CONTROLLER FUNCTIONS       #
    ####################################
    def enable_digital_out(self, output_id: int) -> None:
        """ Enable a digital UR control box output

        Args:
            output_id: Desired output id
        """
        if 0 <= output_id <= 7:
            success = self.rtde_io.setStandardDigitalOut(output_id, True)
        else:
            raise ValueError(f"Desired output id {output_id} not allowed. The digital IO range is between 0 and 7.")

    def disable_digital_out(self, output_id: int) -> None:
        """ Enable a digital UR control box output

            Args:
                output_id: Desired output id
        """
        if 0 <= output_id <= 7:
            success = self.rtde_io.setStandardDigitalOut(output_id, False)
        else:
            raise ValueError(f"Desired output id {output_id} not allowed. The digital IO range is between 0 and 7.")

    def get_digital_out_state(self, output_id: int) -> bool:
        """ Get the status of the UR control box digital output

        Args:
            output_id: Digital output id

        Returns:
            True if output is HIGH; False if output is LOW
        """
        if 0 <= output_id <= 7:
            time.sleep(0.1)
            state: bool = self.rtde_receiver.getDigitalOutState(output_id)
        else:
            raise ValueError(f"Desired output id {output_id} not allowed. The digital IO range is between 0 and 7.")
        return state

    ##################################
    #       RECEIVER FUNCTIONS       #
    ##################################
    def get_joint_pos(self) -> list[float]:
        joint_pos: list[float] = self.joint_pos.tolist()
        return joint_pos

    def get_tcp_offset(self) -> Pose:
        tcp_offset_tr = self.tcp_offset
        tcp_offset = tr2ur_format(tcp_offset_tr)
        return Pose().from_xyz(tcp_offset[:3]).from_axis_angle(tcp_offset[3:])

    def get_tcp_pose(self) -> Pose:
        tcp_pose_tr = self.tcp_pose
        tcp_pose = tr2ur_format(tcp_pose_tr)
        return Pose().from_xyz(tcp_pose[:3]).from_axis_angle(tcp_pose[3:])
