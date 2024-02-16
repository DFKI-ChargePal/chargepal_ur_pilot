from __future__ import annotations
# libs
import time
import logging
import numpy as np
from pathlib import Path
from rigmopy import utils_math as rp_math
from rigmopy import Pose, Vector3d, Vector6d

from ur_control.utils import clip
from ur_control.robots import RealURRobot
from ur_control.utils import tr2ur_format, ur_format2tr

from ur_pilot.config_mdl import Config
from ur_pilot.utils import SpatialPDController

# typing
from typing import Sequence
from numpy import typing as npt

LOGGER = logging.getLogger(__name__)


class URRobot(RealURRobot):

    def __init__(self, ur_control_cfg: Path, ur_pilot_cfg: Config) -> None:
        super().__init__(ur_control_cfg)
        # Constants
        self.dt = 1 / self.cfg.robot.rtde_freq
        self.error_scale_motion_mode = 1.0
        self.force_limit = 0.0
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

    def set_up_force_mode(self, gain: float | None = None, damping: float | None = None) -> None:
        gain_scaling = gain if gain else self.ctrl_cfg.robot.force_mode.gain
        damping_fact = damping if damping else self.ctrl_cfg.robot.force_mode.damping
        self.rtde_controller.zeroFtSensor()
        self.rtde_controller.forceModeSetGainScaling(gain_scaling)
        self.rtde_controller.forceModeSetDamping(damping_fact)

    # NOTE - could be implemented like this in ur_control. In addition, the SpatialPDController
    # would have to be implemented for this. Also, the datatypes have to be changed.
    # It has to be checked how to implement the robot movement via a context.
    def set_up_motion_mode(self,
                           error_scale: float | None = None,
                           force_limit: float | None = None,
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
        Kp = Kp_6 if Kp_6 is not None else self.ctrl_cfg.robot.motion_mode.Kp
        Kd = Kd_6 if Kd_6 is not None else self.ctrl_cfg.robot.motion_mode.Kd
        self._motion_pd = SpatialPDController(Kp_6=Kp, Kd_6=Kd)
        self.set_up_force_mode(gain=ft_gain, damping=ft_damping)

    def motion_mode(self, target: Pose) -> None:
        """ Function to update motion target and let the motion controller keep running

        Args:
            target: Target pose of the TCP
        """
        task_frame = 6 * (0.0, )  # Move w.r.t. robot base
        # Get current Pose
        actual = self.get_tcp_pose()
        # Compute spatial error
        pos_error = target.p - actual.p
        aa_error = np.array(rp_math.quaternion_difference(actual.q, target.q).axis_angle)

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
        ax_error = Vector3d().from_xyz(angle_error * axis_error)
        distance_error = np.clip(pos_error.magnitude, -1.0, 1.0)
        po_error = distance_error * pos_error
        motion_error = Vector6d().from_Vector3d(po_error, ax_error)
        f_net = self.error_scale_motion_mode * self.motion_pd.update(motion_error, self.dt)
        # Clip to maximum forces
        f_net_clip = np.clip(f_net.xyzXYZ, a_min=-self.force_limit, a_max=self.force_limit)
        self.force_mode(
            task_frame=ur_format2tr(task_frame),
            selection_vector=6 * (1,),
            wrench=f_net_clip.tolist(),
            )
        # pose = sm.SE3.Rt(R=sm.UnitQuaternion(target.q.wxyz).SO3(), t=target.p.xyz)
        # self.motion_mode_ur(pose)

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

    # TODO - Add description
    def hybrid_mode(self, pose: Pose, wrench: Vector6d) -> None:
        """

        Args:
            pose: Target pose of the TCP
            wrench: Target wrench w.r.t. the TCP

        Returns:
            None
        """
        task_frame = 6 * (0.0, )  # Move w.r.t. robot bose
        # Get current pose
        cur_pose = self.get_tcp_pose()
        # Compute spatial error
        pos_error = pose.p - cur_pose.p
        aa_error = np.array(rp_math.quaternion_difference(cur_pose.q, pose.q).axis_angle)
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
        ax_error = Vector3d().from_xyz(angle_error * axis_error)
        distance_error = np.clip(pos_error.magnitude, -1.0, 1.0)
        po_error = distance_error * pos_error
        motion_error = Vector6d().from_Vector3d(po_error, ax_error)
        force_error = wrench
        f_net = self.error_scale_motion_mode * (
            self.motion_pd.update(motion_error, self.dt) + self.force_pd.update(force_error, self.dt))
        # Clip to maximum forces
        f_net_clip = np.clip(f_net.xyzXYZ, a_min=-self.force_limit, a_max=self.force_limit)
        self.force_mode(task_frame=ur_format2tr(task_frame), selection_vector=6 * (1,), wrench=f_net_clip.tolist())

    def stop_hybrid_mode(self) -> None:
        """ Function to set robot back in default control mode """
        self.force_pd.reset()
        self.motion_pd.reset()
        self.stop_force_mode()

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
