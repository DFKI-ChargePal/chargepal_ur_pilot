from __future__ import annotations
# global
import time
import logging
import numpy as np
from pathlib import Path
from rigmopy import utils_math as rp_math
from rigmopy import Pose, Quaternion, Transformation, Vector3d, Vector6d

import spatialmath as sm

# local
from ur_pilot import config
from ur_pilot.utils import SpatialPDController
from ur_pilot.rtde_interface import RTDEInterface
from ur_pilot.config_mdl import Config, read_toml, write_toml
from ur_pilot.end_effector.bota_sensor import BotaFtSensor
from ur_pilot.end_effector.flange_eye_calibration import FlangeEyeCalibration
from ur_pilot.end_effector.models import CameraModel, ToolModel, BotaSensONEModel

from ur_control.robots import RealURRobot
from ur_control.utils import tr2ur_format, ur_format2tr

# typing
from typing import Sequence
from numpy import typing as npt
from camera_kit import CameraBase


LOGGER = logging.getLogger(__name__)


class Robot(RealURRobot):

    Q_WORLD2BASE_ = Quaternion().from_euler_angle([np.pi, 0.0, 0.0])

    FT_SENSOR_FRAMES_ = ['world', 'arm_base', 'ft_sensor']

    EE_FRAMES_ = ['flange', 'ft_sensor', 'tool_tip', 'tool_sense', 'camera']

    def __init__(self, cfg_path: Path | None = None) -> None:
        
        if cfg_path is None:
            self.config_fp = Path(config.__file__).parent.joinpath(config.RUNNING_CONFIG_FILE)
            LOGGER.warning(f"No configuration file given. Using default values.")
        else:
            self.config_fp = cfg_path
        config_raw = read_toml(self.config_fp)
        self.pilot_cfg = Config(**config_raw)

        # Constants
        self.error_scale_motion_mode = 1.0
        self.force_limit = 0.0

        # Robot interface
        self.rtde = RTDEInterface(self.cfg.robot.ip_address, self.cfg.robot.rtde_freq, True)
        self.rtde_controller = self.rtde.c
        self.rtde_receiver = self.rtde.r

        # If there is no configuration for the home_position set to current position
        if self.pilot_cfg.robot.home_radians is None:
            self.pilot_cfg.robot.home_radians = list(self.rtde_receiver.getActualQ())

        # Set up end-effector
        if self.pilot_cfg.robot.ft_sensor is None:
            self.extern_sensor = False
            self._ft_sensor = None
            self._ft_sensor_mdl = None
        else:
            self.extern_sensor = True
            self._ft_sensor = BotaFtSensor(**self.pilot_cfg.robot.ft_sensor.dict())
            self._ft_sensor_mdl = BotaSensONEModel()
            self._ft_sensor.start()

        self._motion_pd: SpatialPDController | None = None
        self._force_pd: SpatialPDController | None = None

        self.tool = ToolModel(**self.pilot_cfg.robot.tool.dict())
        self.set_tcp()
        self.cam: CameraBase | None = None
        self.cam_mdl = CameraModel()

    @property
    def home_joint_config(self) -> tuple[float, ...]:
        if self.pilot_cfg.robot.home_radians is not None:
            return tuple(self.pilot_cfg.robot.home_radians)
        else:
            raise ValueError("Home joint configuration was not set.")

    def overwrite_home_joint_config(self, joint_pos: npt.ArrayLike) -> None:
        jp = np.reshape(joint_pos, 6).tolist()
        config_raw = read_toml(self.config_fp)
        config_raw['robot']['home_radians'] = jp
        write_toml(config_raw, self.config_fp)

    @property
    def ft_sensor(self) -> BotaFtSensor:
        if self._ft_sensor is not None:
            return self._ft_sensor
        else:
            raise RuntimeError("External sensor is not initialized. Please check the configuration.")

    @property
    def ft_sensor_mdl(self) -> BotaSensONEModel:
        if self._ft_sensor_mdl is not None:
            return self._ft_sensor_mdl
        else:
            raise RuntimeError("External sensor is not initialized. Please check the configuration.")

    @property
    def motion_pd(self) -> SpatialPDController:
        if self._motion_pd is not None:
            return self._motion_pd
        else:
            raise RuntimeError(
                "Motion PD controller is not initialized. Please run URPilot.set_up_motion_mode(...) first")

    @property
    def force_pd(self) -> SpatialPDController:
        if self._force_pd is None:
            raise RuntimeError(
                "Hybrid PD controller is not initialized. Please run URPilot.set_up_motion_mode(...) first")
        else:
            return self._force_pd

    @property
    def q_world2arm(self) -> Quaternion:
        return self.Q_WORLD2BASE_

    def average_ft_measurement(self, num_meas: int = 100) -> Vector6d:
        """ Method to get an average force torque measurement over num_meas samples.

        Args:
            num_meas: Number of samples

        Returns:
            The mean of n ft measurements
        """
        if num_meas < 1:
            raise ValueError("Number of measurements must be at least 1")
        avg_ft = None
        for _ in range(num_meas):
            ft_next = np.reshape(self.ft_sensor.FT_raw, [6, 1])
            if avg_ft is None:
                avg_ft = ft_next
            else:
                avg_ft = np.hstack([avg_ft, ft_next])
            time.sleep(self.ft_sensor.time_step)
        assert avg_ft is not None  # for type check
        return Vector6d().from_xyzXYZ(np.mean(avg_ft, axis=-1))

    def exit(self) -> None:
        if self._ft_sensor is not None:
            self._ft_sensor.stop()
        self.rtde.exit()

    def register_ee_cam(self, cam: CameraBase, tf_dir: str = "") -> None:
        self.cam = cam
        if not self.cam.is_calibrated:
            self.cam.load_coefficients()
        T_flange2cam = FlangeEyeCalibration.load_transformation(self.cam, tf_dir)
        self.cam_mdl.T_flange2camera = Transformation().from_trans_matrix(T_flange2cam)

    ####################################
    #       CONTROLLER FUNCTIONS       #
    ####################################
    # def move_home(self) -> None:
    #     LOGGER.debug("Try to move robot in home configuration")
    #     success = self.rtde.c.moveJ(self.cfg.robot.home_radians, self.cfg.robot.joints.vel, self.cfg.robot.joints.acc)
    #     if not success:
    #         LOGGER.warning("Malfunction during movement to the home configuration!")

    def move_l(self, tcp_pose: Pose) -> None:
        quat = tcp_pose.q.wxyz
        pos = tcp_pose.p.xyz
        pose = sm.SE3.Rt(R=sm.UnitQuaternion(quat).SO3(), t=pos)
        self.movel(pose)

    def move_j(self, q: Sequence[float]) -> None:
        self.movej(q)

    def servo_l(self, target: Pose) -> None:
        quat = target.q.wxyz
        pos = target.p.xyz
        pose = sm.SE3.Rt(R=sm.UnitQuaternion(quat).SO3(), t=pos)
        self.servol(pose)

    def stop_servoing(self) -> None:
        self.servo_stop()

    # def set_up_force_mode(self, gain: float | None = None, damping: float | None = None) -> None:
    #     gain_scaling = gain if gain else self.cfg.robot.force_mode.gain
    #     damping_fact = damping if damping else self.cfg.robot.force_mode.damping
    #     self.rtde.c.zeroFtSensor()
    #     self.rtde.c.forceModeSetGainScaling(gain_scaling)
    #     self.rtde.c.forceModeSetDamping(damping_fact)

    #NOTE - use spatialmath.SE3 for the task frame?
    # def force_mode(self,
    #                task_frame: Sequence[float],
    #                selection_vector: Sequence[int],
    #                wrench: Sequence[float],
    #                f_mode_type: int | None = None,
    #                tcp_limits: Sequence[float] | None = None
    #                ) -> None:
    #     """ Function to use the force mode of the ur_rtde API """
    #     if f_mode_type is None:
    #         f_mode_type = self.cfg.robot.force_mode.mode
    #     if tcp_limits is None:
    #         tcp_limits = self.cfg.robot.force_mode.tcp_speed_limits
    #     self.rtde.c.forceMode(
    #         task_frame,
    #         selection_vector,
    #         wrench,
    #         f_mode_type,
    #         tcp_limits
    #     )
    #     time.sleep(self.rtde.dt)

    # def stop_force_mode(self) -> None:
    #     """ Function to set robot back in normal position control mode. """
    #     self.rtde.c.forceModeStop()

    #NOTE - could be implemented like this in ur_control. In addition, the SpatialPDController 
    # would have to be implemented for this. Also the datatypes have to be changed.
    # It has to be checked how to implement the robot movement via a context.
    # def set_up_motion_mode(self, 
    #                        error_scale: float | None = None,
    #                        force_limit: float | None = None,
    #                        Kp_6: Sequence[float] | None = None,
    #                        Kd_6: Sequence[float] | None = None, 
    #                        ft_gain: float | None = None, 
    #                        ft_damping: float | None = None) -> None:
    #     """ Function to set up force based motion controller

    #     Args:
    #         error_scale: Overall scaling parameter
    #         force_limit: The absolute value of the applied forces in tool space
    #         Kp_6: 6 dimensional motion controller proportional gain
    #         Kd_6: 6 dimensional motion controller derivative gain
    #         ft_gain: Force torque gain parameter
    #         ft_damping: Force torque damping parameter
    #     """
    #     self.error_scale_motion_mode = error_scale if error_scale else self.cfg.robot.motion_mode.error_scale
    #     self.force_limit = force_limit if force_limit else self.cfg.robot.motion_mode.force_limit
    #     Kp = Kp_6 if Kp_6 is not None else self.cfg.robot.motion_mode.Kp
    #     Kd = Kd_6 if Kd_6 is not None else self.cfg.robot.motion_mode.Kd
    #     self._motion_pd = SpatialPDController(Kp_6=Kp, Kd_6=Kd)
    #     self.set_up_force_mode(gain=ft_gain, damping=ft_damping)

    # def motion_mode(self, target: Pose) -> None:
    #     """ Function to update motion target and let the motion controller keep running

    #     Args:
    #         target: Target pose of the TCP
    #     """
    #     task_frame = 6 * (0.0, )  # Move w.r.t. robot base
    #     # Get current Pose
    #     actual = self.get_tcp_pose()
    #     # Compute spatial error
    #     pos_error = target.p - actual.p
    #     aa_error = np.array(rp_math.quaternion_difference(actual.q, target.q).axis_angle)

    #     # Angles error always within [0,Pi)
    #     angle_error = np.max(np.abs(aa_error))
    #     if angle_error < 1e7:
    #         axis_error = aa_error
    #     else:
    #         axis_error = aa_error/angle_error
    #     # Clamp maximal tolerated error.
    #     # The remaining error will be handled in the next control cycle.
    #     # Note that this is also the maximal offset that the
    #     # cartesian_compliance_controller can use to build up a restoring stiffness
    #     # wrench.
    #     angle_error = np.clip(angle_error, 0.0, 1.0)
    #     ax_error = Vector3d().from_xyz(angle_error * axis_error)
    #     distance_error = np.clip(pos_error.magnitude, -1.0, 1.0)
    #     po_error = distance_error * pos_error
    #     motion_error = Vector6d().from_Vector3d(po_error, ax_error)
    #     f_net = self.error_scale_motion_mode * self.motion_pd.update(motion_error, self.rtde.dt)
    #     # Clip to maximum forces
    #     f_net_clip = np.clip(f_net.xyzXYZ, a_min=-self.force_limit, a_max=self.force_limit)
    #     self.force_mode(
    #         task_frame=task_frame,
    #         selection_vector=6 * (1,),
    #         wrench=f_net_clip.tolist(),
    #         )
        
        # pose = sm.SE3.Rt(R=sm.UnitQuaternion(target.q.wxyz).SO3(), t=target.p.xyz)
        # self.motion_mode_ur(pose)

    # def stop_motion_mode(self) -> None:
    #     """ Function to set robot back in normal position control mode. """
    #     self.motion_pd.reset()
    #     self.stop_force_mode()

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
        self.error_scale_motion_mode = error_scale if error_scale else self.cfg.robot.hybrid_mode.error_scale
        self.force_limit = force_limit if force_limit else self.cfg.robot.hybrid_mode.force_limit
        f_Kp = Kp_6_force if Kp_6_force is not None else self.cfg.robot.hybrid_mode.Kp_force
        f_Kd = Kd_6_force if Kd_6_force is not None else self.cfg.robot.hybrid_mode.Kd_force
        m_Kp = Kp_6_motion if Kp_6_motion is not None else self.cfg.robot.hybrid_mode.Kp_motion
        m_Kd = Kd_6_motion if Kd_6_motion is not None else self.cfg.robot.hybrid_mode.Kd_motion
        self._force_pd = SpatialPDController(Kp_6=f_Kp, Kd_6=f_Kd)
        self._motion_pd = SpatialPDController(Kp_6=m_Kp, Kd_6=m_Kd)
        self.set_up_force_mode(gain=ft_gain, damping=ft_damping)

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
            self.motion_pd.update(motion_error, self.rtde.dt) + self.force_pd.update(force_error, self.rtde.dt))
        # Clip to maximum forces
        f_net_clip = np.clip(f_net.xyzXYZ, a_min=-self.force_limit, a_max=self.force_limit)
        self.force_mode(task_frame=task_frame, selection_vector=6 * (1, ), wrench=f_net_clip.tolist())

    def stop_hybrid_mode(self) -> None:
        """ Function to set robot back in default control mode """
        self.force_pd.reset()
        self.motion_pd.reset()
        self.stop_force_mode()

    # def set_up_teach_mode(self) -> None:
    #     """ Function to enable the free drive mode. """
    #     self.rtde.c.teachMode()

    # def stop_teach_mode(self) -> None:
    #     """ Function to set robot back in normal position control mode. """
    #     self.rtde.c.endTeachMode()

    def set_tcp(self, frame: str = 'tool_tip') -> None:
        """ Function to set the tcp relative to the tool flange. """
        offset = self.__frame_to_offset(frame=frame)
        self.rtde.c.setTcp(offset)
        LOGGER.info(f"From now on robot end effector is going to work with respect to frame: {frame}")

    ##################################
    #       RECEIVER FUNCTIONS       #
    ##################################
    def get_joint_pos(self) -> list[float]:
    #     joint_pos: list[float] = self.rtde.r.getActualQ()
    #     return joint_pos
        return self.joint_pos

    def get_joint_vel(self) -> Sequence[float]:
        # joint_vel: Sequence[float] = self.rtde.r.getActualQd()
        # return joint_vel
        return self.joint_vel

    def get_pose(self, frame: str = 'flange') -> Pose:
        """ Get pose of the desired frame w.r.t the robot base. This function is independent of the TCP offset defined
            on the robot side.
        Args:
            frame: One of the frame name defined in the class member variable 'EE_FRAMES_'
        Returns:
            6D pose
        """
        tcp_offset = self.__frame_to_offset(frame=frame)
        q: Sequence[float] = self.rtde.r.getActualQ()
        pose_vec: Sequence[float] = self.rtde.c.getForwardKinematics(q=q, tcp_offset=tcp_offset)
        return Pose().from_xyz(pose_vec[:3]).from_axis_angle(pose_vec[3:])

    def __frame_to_offset(self, frame: str) -> tuple[float, ...]:
        if frame not in self.EE_FRAMES_:
            raise ValueError(f"Given frame is not available. Please select one of '{self.EE_FRAMES_}'")
        if frame == 'flange':
            offset = 6 * (0.0,)
        elif frame == 'ft_sensor':
            if self.extern_sensor:
                offset = self.ft_sensor_mdl.p_mounting2wrench.xyz + self.ft_sensor_mdl.q_mounting2wrench.axis_angle
            else:
                # Tread internal force torque sensor as mounted in flange frame
                offset = 6 * (0.0,)
                LOGGER.warning(f"External ft_sensor is not configured. Return pose of the flange frame.")
        elif frame == 'tool_tip':
            if self.extern_sensor:
                pq_flange2tool = (self.ft_sensor_mdl.T_mounting2wrench @ self.tool.T_mounting2tip).pose
                offset = pq_flange2tool.xyz + pq_flange2tool.axis_angle
            else:
                offset = self.tool.tip_frame.xyz + self.tool.tip_frame.axis_angle
        elif frame == 'tool_sense':
            if self.extern_sensor:
                pq_flange2sense = (self.ft_sensor_mdl.T_mounting2wrench @ self.tool.T_mounting2sense).pose
                offset = pq_flange2sense.xyz + pq_flange2sense.axis_angle
            else:
                offset = self.tool.sense_frame.xyz + self.tool.sense_frame.axis_angle
        elif frame == 'camera':
            pq_flange2cam = self.cam_mdl.T_flange2camera.pose
            offset = pq_flange2cam.xyz + pq_flange2cam.axis_angle
        else:
            raise RuntimeError(f"This code should not be reached. Please check the frame definitions.")
        return offset

    def get_tcp_offset(self) -> Pose:
        # tcp_offset: Sequence[float] = self.rtde.c.getTCPOffset()
        tcp_offset_tr = self.tcp_offset
        tcp_offset = tr2ur_format(tcp_offset_tr)
        return Pose().from_xyz(tcp_offset[:3]).from_axis_angle(tcp_offset[3:])

    def get_tcp_pose(self) -> Pose:
        # tcp_pose: Sequence[float] = self.rtde.r.getActualTCPPose()
        tcp_pose_tr = self.tcp_pose
        tcp_pose = tr2ur_format(tcp_pose_tr)
        return Pose().from_xyz(tcp_pose[:3]).from_axis_angle(tcp_pose[3:])

    def get_tcp_vel(self) -> Vector6d:
        # tcp_vel: Sequence[float] = self.rtde.r.getActualTCPSpeed()
        tcp_vel = self.tcp_twist.A
        return Vector6d().from_xyzXYZ(tcp_vel)

    def get_ft(self, frame: str = 'ft_sensor') -> Vector6d:
        """ Get force-torque readings w.r.t the desired frame.

        Args:
            frame: One of the frame name defined in the class member variable 'FT_SENSOR_FRAMES_'

        Returns:
            A 6d vector with the sensor readings
        """
        if frame not in self.FT_SENSOR_FRAMES_:
            raise ValueError(f"Given frame is not available. Please select one of '{self.FT_SENSOR_FRAMES_}'")
        # The default build in sensor readings are w.r.t the arm base frame
        ft = Vector6d()
        if self.extern_sensor:
            # The default sensor readings are w.r.t the sensor frame
            ft_raw = Vector6d().from_xyzXYZ(self.ft_sensor.FT)
            if frame == 'ft_sensor':
                ft = ft_raw
        return ft

    def get_tcp_force(self) -> Vector6d:
        if self.extern_sensor:
            tcp_force = Vector6d().from_xyzXYZ(self.ft_sensor.FT)
        else:
            tcp_force = Vector6d().from_xyzXYZ(self.rtde.r.getActualTCPForce())
        # Compensate Tool mass
        f_tool_wrt_world = self.tool.f_inertia
        f_tool_wrt_ft = (self.q_world2arm * self.get_tcp_pose().q).apply(f_tool_wrt_world, inverse=True)
        t_tool_wrt_ft = Vector3d().from_xyz(np.cross(self.tool.com.xyz, f_tool_wrt_ft.xyz))
        ft_comp = Vector6d().from_Vector3d(f_tool_wrt_ft, t_tool_wrt_ft)
        return tcp_force + ft_comp
