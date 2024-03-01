from __future__ import annotations

# libs
import time
import logging
import numpy as np
import spatialmath as sm
from pathlib import Path

from ur_control.utils import clip, tr2ur_format, ur_format2tr
from ur_control.robots import RealURRobot

from ur_pilot.config_mdl import Config
from ur_pilot.utils import SpatialPDController, vec_to_str

# typing
from typing import Sequence
from numpy import typing as npt

LOGGER = logging.getLogger(__name__)


class URRobot(RealURRobot):

    def __init__(self, ur_control_cfg: Path, ur_pilot_cfg: Config) -> None:
        super().__init__(ur_control_cfg)
        # Constants
        self.dt = 1 / self.cfg.robot_dir.rtde_freq
        self.error_scale_motion_mode = 1.0
        self.force_limit = 0.0
        self.torque_limit = 0.0
        # Control attributes
        self.ctrl_cfg = ur_pilot_cfg
        self._motion_pd: SpatialPDController | None = None
        self._force_pd: SpatialPDController | None = None

    @property
    def force_pd(self) -> SpatialPDController:
        if self._force_pd is None:
            raise RuntimeError(
                "Hybrid PD controller is not initialized. Please run URPilot.set_up_motion_mode(...) first")
        else:
            return self._force_pd

    @property
    def motion_pd(self) -> SpatialPDController:
        if self._motion_pd is not None:
            return self._motion_pd
        else:
            raise RuntimeError(
                "Motion PD controller is not initialized. Please run URPilot.set_up_motion_mode(...) first")

    def set_up_motion_mode(self,
                           error_scale: float | None = None,
                           force_limit: float | None = None,
                           torque_limit: float | None = None,
                           Kp_6: Sequence[float] | None = None,
                           Kd_6: Sequence[float] | None = None,
                           ft_gain: float | None = None,
                           ft_damping: float | None = None) -> None:
        """ Function to set up force based motion controller

        Args:
            error_scale: Overall scaling parameter
            force_limit: The absolute value of the applied forces in tool space
            Kp_6: 6 dimensional motion controller proportional gain
            Kd_6: 6 dimensional motion controller derivative gain
            ft_gain: Force torque gain parameter
            ft_damping: Force torque damping parameter
        """
        self.error_scale_motion_mode = error_scale if error_scale else self.ctrl_cfg.robot.motion_mode.error_scale
        self.force_limit = force_limit if force_limit else self.ctrl_cfg.robot.motion_mode.force_limit
        self.torque_limit = torque_limit if torque_limit else self.ctrl_cfg.robot.motion_mode.torque_limit
        Kp = Kp_6 if Kp_6 is not None else self.ctrl_cfg.robot.motion_mode.Kp
        Kd = Kd_6 if Kd_6 is not None else self.ctrl_cfg.robot.motion_mode.Kd
        self._motion_pd = SpatialPDController(Kp_6=Kp, Kd_6=Kd)
        self.set_up_force_mode(gain=ft_gain, damping=ft_damping)

    def motion_mode(self, target: sm.SE3) -> None:
        """ Function to update motion target and let the motion controller keep running

        Args:
            target: Target pose of the TCP
        """
        task_frame = sm.SE3()  # Move w.r.t. robot base
        # Compute spatial error
        T_31 = target
        T_21 = self.tcp_pose
        T_12 = T_21.inv()
        T_32 = T_31 * T_12
        pos_error = T_32.t
        aa_error = T_32.eulervec()

        # Angles error always within [0,Pi)
        angle_error = np.max(np.abs(aa_error))
        if angle_error < 1e7:
            axis_error = aa_error
        else:
            axis_error = aa_error/angle_error
        # Clamp maximal tolerated error.
        # The remaining error will be handled in the next control cycle.
        # Note that this is also the maximal offset that the
        # cartesian_compliance_controller can use to build up a restoring stiffness
        # wrench.
        angle_error = np.clip(angle_error, 0.0, 1.0)
        ax_error = angle_error * axis_error
        distance_error = np.clip(np.linalg.norm(pos_error), -1.0, 1.0)
        po_error = distance_error * pos_error
        motion_error = np.append(po_error, ax_error)
        f_net = self.error_scale_motion_mode * self.motion_pd.update(motion_error, self.dt)
        # Clip to maximum forces
        f_net_clip = np.append(
            np.clip(f_net[0:3], a_min=-self.force_limit, a_max=self.force_limit), 
            np.clip(f_net[3:6], a_min=-self.torque_limit, a_max=self.torque_limit)
            )
        self.force_mode(task_frame=task_frame, selection_vector=6 * (1,), wrench=f_net_clip.tolist())

    def stop_motion_mode(self) -> None:
        """ Function to set robot back in normal position control mode. """
        self.motion_pd.reset()
        self.stop_force_mode()

    def set_up_hybrid_mode(self,
                           error_scale: float | None = None,
                           force_limit: float | None = None,
                           Kp_6_force: Sequence[float] | None = None,
                           Kd_6_force: Sequence[float] | None = None,
                           Kp_6_motion: Sequence[float] | None = None,
                           Kd_6_motion: Sequence[float] | None = None,
                           ft_gain: float | None = None,
                           ft_damping: float | None = None) -> None:
        """ Function to set up the hybrid controller. Error signal is a combination of a pose and wrench error

        Args:
            error_scale: Overall error scaling parameter
            force_limit: The absolute value of the applied forces in tool space
            Kp_6_force: 6 dimensional controller proportional gain (force part)
            Kd_6_force: 6 dimensional controller derivative gain (force part)
            Kp_6_motion: 6 dimensional controller proportional gain (motion part)
            Kd_6_motion: 6 dimensional controller derivative gain (motion part)
            ft_gain: Force torque gain parameter (low level controller)
            ft_damping: Force torque damping parameter (low level controller)

        Returns:
            None
        """
        self.error_scale_motion_mode = error_scale if error_scale else self.ctrl_cfg.robot.hybrid_mode.error_scale
        self.force_limit = force_limit if force_limit else self.ctrl_cfg.robot.hybrid_mode.force_limit
        f_Kp = Kp_6_force if Kp_6_force is not None else self.ctrl_cfg.robot.hybrid_mode.Kp_force
        f_Kd = Kd_6_force if Kd_6_force is not None else self.ctrl_cfg.robot.hybrid_mode.Kd_force
        m_Kp = Kp_6_motion if Kp_6_motion is not None else self.ctrl_cfg.robot.hybrid_mode.Kp_motion
        m_Kd = Kd_6_motion if Kd_6_motion is not None else self.ctrl_cfg.robot.hybrid_mode.Kd_motion
        self._force_pd = SpatialPDController(Kp_6=f_Kp, Kd_6=f_Kd)
        self._motion_pd = SpatialPDController(Kp_6=m_Kp, Kd_6=m_Kd)
        self.set_up_force_mode(gain=ft_gain, damping=ft_damping)

    def hybrid_mode(self, target: sm.SE3, wrench: npt.ArrayLike) -> None:
        """ Update hybrid mode control error

        Args:
            target:   Target pose of the TCP
            wrench: Target wrench w.r.t. the TCP

        """
        task_frame = sm.SE3()  # Move w.r.t. robot base
        # Compute spatial error
        T_31 = target
        T_21 = self.tcp_pose
        T_12 = T_21.inv()
        T_32 = T_31 * T_12
        pos_error = T_32.t
        aa_error = T_32.eulervec()

        # Angles error always within [0,Pi)
        angle_error = np.max(np.abs(aa_error))
        if angle_error < 1e7:
            axis_error = aa_error
        else:
            axis_error = aa_error/angle_error
        # Clamp maximal tolerated error.
        # The remaining error will be handled in the next control cycle.
        # Note that this is also the maximal offset that the
        # cartesian_compliance_controller can use to build up a restoring stiffness
        # wrench.
        angle_error = np.clip(angle_error, 0.0, 1.0)
        ax_error = angle_error * axis_error
        distance_error = np.clip(pos_error.magnitude, -1.0, 1.0)
        po_error = distance_error * pos_error
        motion_error = np.append(po_error, ax_error)
        force_error = np.reshape(wrench, 6)
        f_net = self.error_scale_motion_mode * (
            self.motion_pd.update(motion_error, self.dt) + self.force_pd.update(force_error, self.dt))
        # Clip to maximum forces
        f_net_clip = np.clip(f_net, a_min=-self.force_limit, a_max=self.force_limit)
        self.force_mode(task_frame=task_frame, selection_vector=6 * (1,), wrench=f_net_clip.tolist())

    def stop_hybrid_mode(self) -> None:
        """ Function to set robot back in default control mode """
        self.force_pd.reset()
        self.motion_pd.reset()
        self.stop_force_mode()

    def move_path_j(self,
                    wps: Sequence[npt.ArrayLike],
                    vel: float | None = None,
                    acc: float | None = None,
                    bf: float | None = None) -> bool:
        """ Move the robot in joint space along a specified path.

        Args:
            wps: Waypoints of the path. List of arrays of the target joint angles in radians
            vel: Joint speed of leading axis \(^{rad}/_{s^2}\). Defaults to None.
            acc: Joint acceleration of leading axis \(^{rad}/{s^2}\). Defaults to None.
            bf:  Blend factor to smooth movements.
        """
        wps_f32 = [np.array(np.reshape(target, 6), dtype=np.float32) for target in wps]
        speed = (
            clip(vel, 0.0, self.cfg.robot_dir.joints.max_vel)
            if vel
            else self.cfg.robot_dir.joints.vel
        )
        acceleration = (
            clip(acc, 0.0, self.cfg.robot_dir.joints.max_acc)
            if acc
            else self.cfg.robot_dir.joints.acc
        )
        bf = clip(bf, 0.0, 0.1) if bf else 0.02
        path = [[*tg.tolist(), speed, acceleration, bf] for tg in wps_f32]
        # Set blend factor of the last waypoint to zero to stop smoothly
        path[-1][-1] = 0.0
        success: bool = self.rtde_controller.moveJ(path=path)
        return success

    ####################################
    # --- #   HELPER FUNCTIONS   # --- #
    ####################################
    def enable_digital_out(self, output_id: int) -> bool:
        """ Enable a digital UR control box output

        Args:
            output_id: Desired output id
        """
        if 0 <= output_id <= 7:
            success: bool = self.rtde_io.setStandardDigitalOut(output_id, True)
        else:
            raise ValueError(f"Desired output id {output_id} not allowed. The digital IO range is between 0 and 7.")
        return success

    def disable_digital_out(self, output_id: int) -> bool:
        """ Enable a digital UR control box output

        Args:
            output_id: Desired output id
        """
        if 0 <= output_id <= 7:
            success: bool = self.rtde_io.setStandardDigitalOut(output_id, False)
        else:
            raise ValueError(f"Desired output id {output_id} not allowed. The digital IO range is between 0 and 7.")
        return success

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

    def get_pose(self, offset: sm.SE3) -> sm.SE3:
        """ Get robot end-effector pose modified by offset
        Args:
            offset: The offset transformation w.r.t. robot flange

        Returns:
            End-effector pose with offset as SE(3) transformation matrix
        """
        ur_pose_vec = self.rtde_controller.getForwardKinematics(q=self.joint_pos, tcp_offset=tr2ur_format(offset))
        return ur_format2tr(ur_pose_vec)
