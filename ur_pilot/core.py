from __future__ import annotations
# global
import time
import logging
import numpy as np
from pathlib import Path
import chargepal_aruco as ca
from chargepal_aruco import Camera
from rigmopy import Pose, Transformation, Vector6d

# local
import config
from ur_pilot.rtde_interface import RTDEInterface
from ur_pilot.config import Config, read_toml

# typing
from typing import Sequence


LOGGER = logging.getLogger(__name__)


class URPilot:

    def __init__(self) -> None:

        config_fp = Path(config.__file__).parent.joinpath(config.RUNNING_CONFIG_FILE)
        config_dict = read_toml(config_fp)
        self.cfg = Config(**config_dict)

        # Robot interface
        self.rtde = RTDEInterface(self.cfg.robot.ip_address, self.cfg.robot.rtde_freq, True)

        # If there is no configuration for the home_position set to current position
        if self.cfg.robot.home_radians is None:
            self.cfg.robot.home_radians = list(self.rtde.r.getActualQ())

        # End-effector camera
        self.cam: Camera | None = None
        self.T_tcp2cam = Transformation()

    @property
    def home_joint_config(self) -> tuple[float, ...]:
        if self.cfg.robot.home_radians is not None:
            return tuple(self.cfg.robot.home_radians)
        else:
            raise ValueError("Home joint configuration was not set.")

    def exit(self) -> None:
        self.rtde.exit()

    def register_ee_cam(self, cam: Camera) -> None:
        self.cam = cam
        self.cam.load_coefficients()
        T_tcp2cam = ca.Calibration.hand_eye_calibration_load_transformation(self.cam)
        self.T_tcp2cam = Transformation().from_trans_matrix(T_tcp2cam)

    ####################################
    #       CONTROLLER FUNCTIONS       #
    ####################################
    def move_home(self) -> None:
        LOGGER.debug("Try to move robot in home configuration")
        success = self.rtde.c.moveJ(self.cfg.robot.home_radians, self.cfg.robot.joints.vel, self.cfg.robot.joints.acc)
        if not success:
            LOGGER.warning("Malfunction during movement to the home configuration!")

    def move_l(self, tcp_pose: Pose) -> None:
        LOGGER.debug(f"Try to move robot to TCP pose {tcp_pose}")
        success = self.rtde.c.moveL(
            pose=tcp_pose.xyz + tcp_pose.axis_angle,
            speed=self.cfg.robot.tcp.vel,
            acceleration=self.cfg.robot.tcp.acc
        )
        if not success:
            cur_pose = self.rtde.r.getActualTCPPose()
            tgt_msg = f"\nTarget pose: {tcp_pose}"
            cur_msg = f"\nCurrent pose: {cur_pose}"
            LOGGER.warning(f"Malfunction during movement to new pose.{tgt_msg}{cur_msg}")

    def move_j(self, q: Sequence[float]) -> None:
        LOGGER.debug(f"Try to move the robot to new joint configuration {q}")
        success = self.rtde.c.moveJ(q, self.cfg.robot.joints.vel, self.cfg.robot.joints.acc)
        if not success:
            cur_q = self.rtde.r.getActualQ()
            tgt_msg = f"\nTarget joint positions: {q}"
            cur_msg = f"\nCurrent joint positions: {cur_q}"
            LOGGER.warning(f"Malfunction during movement to new joint positions.{tgt_msg}{cur_msg}")

    def servo_l(self, tcp_pose: Pose) -> None:
        LOGGER.debug(f"Try to move robot to TCP pose {tcp_pose}")
        self.rtde.c.initPeriod()
        success = self.rtde.c.servoL(
            tcp_pose.xyz + tcp_pose.axis_angle,
            self.cfg.robot.servo.vel,
            self.cfg.robot.servo.acc,
            self.rtde.dt,
            self.cfg.robot.servo.lkh_time,
            self.cfg.robot.servo.gain
        )
        self.rtde.c.waitPeriod(self.rtde.dt)
        # Since there is no real time kernel at the moment use python time library
        time.sleep(self.rtde.dt)  # Do we need this?
        if not success:
            cur_pose = self.rtde.r.getActualTCPPose()
            tgt_msg = f"\nTarget pose: {tcp_pose}"
            cur_msg = f"\nCurrent pose: {cur_pose}"
            LOGGER.warning(f"Malfunction during movement to new pose.{tgt_msg}{cur_msg}")

    def stop_servoing(self) -> None:
        self.rtde.c.servoStop()

    def set_up_force_mode(self, gain: float | None = None, damping: float | None = None) -> None:
        gain_scaling = gain if gain else self.cfg.robot.force_mode.gain
        damping_fact = damping if damping else self.cfg.robot.force_mode.damping
        self.rtde.c.zeroFtSensor()
        self.rtde.c.forceModeSetGainScaling(gain_scaling)
        self.rtde.c.forceModeSetDamping(damping_fact)

    def force_mode(self,
                   task_frame: Sequence[float],
                   selection_vector: Sequence[int],
                   wrench: Sequence[float],
                   f_mode_type: int | None = None,
                   tcp_limits: Sequence[float] | None = None
                   ) -> None:
        """ Function to use the force mode of the ur_rtde API """
        if f_mode_type is None:
            f_mode_type = self.cfg.robot.force_mode.mode
        if tcp_limits is None:
            tcp_limits = self.cfg.robot.force_mode.tcp_speed_limits
        self.rtde.c.forceMode(
            task_frame,
            selection_vector,
            wrench,
            f_mode_type,
            tcp_limits
        )
        time.sleep(self.rtde.dt)

    def stop_force_mode(self) -> None:
        """ Function to set robot back in normal position control mode. """
        self.rtde.c.forceModeStop()

    def teach_mode(self) -> None:
        """ Function to enable the free drive mode. """
        self.rtde.c.teachMode()

    def stop_teach_mode(self) -> None:
        """ Function to set robot back in normal position control mode. """
        self.rtde.c.endTeachMode()

    def set_tcp(self, tcp_offset: Pose) -> None:
        """ Function to set the tcp relative to the tool flange. """
        self.rtde.c.setTcp(tcp_offset.xyz + tcp_offset.axis_angle)

    ##################################
    #       RECEIVER FUNCTIONS       #
    ##################################
    def get_joint_pos(self) -> Sequence[float]:
        joint_pos: Sequence[float] = self.rtde.r.getActualQ()
        return joint_pos

    def get_joint_vel(self) -> Sequence[float]:
        joint_vel: Sequence[float] = self.rtde.r.getActualQd()
        return joint_vel

    def get_tcp_offset(self) -> Pose:
        tcp_offset: Sequence[float] = self.rtde.c.getTCPOffset()
        return Pose().from_xyz(tcp_offset[:3]).from_axis_angle(tcp_offset[3:])

    def get_tcp_pose(self) -> Pose:
        tcp_pose: Sequence[float] = self.rtde.r.getActualTCPPose()
        return Pose().from_xyz(tcp_pose[:3]).from_axis_angle(tcp_pose[3:])

    def get_tcp_force(self) -> Vector6d:
        tcp_force: Sequence[float] = self.rtde.r.getActualTCPForce()
        return Vector6d().from_xyzXYZ(tcp_force)
