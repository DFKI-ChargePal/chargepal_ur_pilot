from __future__ import annotations
# global
import time
import logging
import numpy as np
from pathlib import Path
import chargepal_aruco as ca
from chargepal_aruco import Camera
from rigmopy import utils_math as rp_math
from rigmopy import Pose, Quaternion, Transformation, Vector3d, Vector6d

# local
import config
from ur_pilot.utils import SpatialPDController
from ur_pilot.rtde_interface import RTDEInterface
from ur_pilot.config_mdl import Config, read_toml
from ur_pilot.end_effector.models import CameraModel, ToolModel, BotaSensONEModel
from ur_pilot.end_effector.bota_sensor import BotaFtSensor

# typing
from typing import Sequence


LOGGER = logging.getLogger(__name__)


class URPilot:

    Q_WORLD2BASE_ = Quaternion().from_euler_angle([np.pi, 0.0, 0.0])

    FT_SENSOR_FRAMES_ = ['world', 'arm_base', 'ft_sensor']

    EE_FRAMES_ = ['flange', 'ft_sensor', 'tool', 'camera']

    def __init__(self) -> None:

        config_fp = Path(config.__file__).parent.joinpath(config.RUNNING_CONFIG_FILE)
        config_dict = read_toml(config_fp)
        self.cfg = Config(**config_dict)

        # Constants
        self.error_scale_motion_mode = 1.0

        # Robot interface
        self.rtde = RTDEInterface(self.cfg.robot.ip_address, self.cfg.robot.rtde_freq, True)

        # If there is no configuration for the home_position set to current position
        if self.cfg.robot.home_radians is None:
            self.cfg.robot.home_radians = list(self.rtde.r.getActualQ())

        # Set up end-effector
        if self.cfg.robot.ft_sensor is None:
            self.extern_sensor = False
            self._ft_sensor = None
            self._ft_sensor_mdl = None
        else:
            self.extern_sensor = True
            self._ft_sensor = BotaFtSensor(**self.cfg.robot.ft_sensor.dict())
            self._ft_sensor_mdl = BotaSensONEModel()
            self._ft_sensor.start()

        self._motion_pd: SpatialPDController | None = None

        self.tool = ToolModel(**self.cfg.robot.tool.dict())
        self.cam: Camera | None = None
        self.cam_mdl = CameraModel()

    @property
    def home_joint_config(self) -> tuple[float, ...]:
        if self.cfg.robot.home_radians is not None:
            return tuple(self.cfg.robot.home_radians)
        else:
            raise ValueError("Home joint configuration was not set.")

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

    def register_ee_cam(self, cam: Camera) -> None:
        self.cam = cam
        if not self.cam.is_calibrated:
            self.cam.load_coefficients()
        T_tcp2cam = ca.Calibration.hand_eye_calibration_load_transformation(self.cam)
        self.cam_mdl.T_flange2camera = Transformation().from_trans_matrix(T_tcp2cam)

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

    def servo_l(self, target: Pose) -> None:
        LOGGER.debug(f"Try to move robot to TCP pose {target}")
        self.rtde.c.initPeriod()
        success = self.rtde.c.servoL(
            target.xyz + target.axis_angle,
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
            tgt_msg = f"\nTarget pose: {target}"
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

    def set_up_motion_mode(self, 
                           error_scale: float | None = None,
                           Kp_6: Sequence[float] | None = None,
                           Kd_6: Sequence[float] | None = None, 
                           ft_gain: float | None = None, 
                           ft_damping: float | None = None) -> None:
        """ Function to set up force based motion controller

        Args:
            ft_gain: _description_. Defaults to None.
            ft_damping: _description_. Defaults to None.
        """
        self.error_scale_motion_mode = error_scale if error_scale else self.cfg.robot.motion_mode.error_scale
        Kp = Kp_6 if Kp_6 is not None else self.cfg.robot.motion_mode.Kp
        Kd = Kd_6 if Kd_6 is not None else self.cfg.robot.motion_mode.Kd
        self._motion_pd = SpatialPDController(Kp_6=Kp, Kd_6=Kd)
        self.set_up_force_mode(gain=ft_gain, damping=ft_damping)

    def motion_mode(self, target: Pose) -> None:
        """ Function to update motion target and let the motion controller keep running

        Args:
            tcp_pose: Target pose of the TCP
        """
        task_frame = 6 * (0.0, )  # Move wrt. robot base?
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
        f_net = self.error_scale_motion_mode * self.motion_pd.update(motion_error, self.rtde.dt)
        # Clip to maximum forces
        f_net_clip = np.clip(f_net.xyzXYZ, a_min=-100, a_max=100)
        self.force_mode(
            task_frame=task_frame,
            selection_vector=6 * (1,),
            wrench=f_net_clip.tolist(),
            )

    def stop_motion_mode(self) -> None:
        """ Function to set robot back in normal position control mode. """
        self.motion_pd.reset()
        self.stop_force_mode()

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

    def get_pose(self, frame: str = 'flange') -> Pose:
        """ Get pose of the desired frame w.r.t the robot base. This function is independent of the TCP offset defined
            on the robot side.

        Args:
            frame: One of the frame name defined in the class member variable 'EE_FRAMES_'

        Returns:
            6D pose
        """
        if frame not in self.EE_FRAMES_:
            raise ValueError(f"Given frame is not available. Please select one of '{self.EE_FRAMES_}'")

        if frame == 'flange':
            tcp_offset_ = 6 * (0.0,)
        elif frame == 'ft_sensor':
            if self.extern_sensor:
                tcp_offset_ = self.ft_sensor_mdl.p_mounting2wrench.xyz + self.ft_sensor_mdl.q_mounting2wrench.axis_angle
            else:
                # Tread internal force torque sensor as mounted in flange frame
                tcp_offset_ = 6 * (0.0,)
                LOGGER.warning(f"External ft_sensor is not configured. Return pose of the flange frame.")
        elif frame == 'tool':
            if self.extern_sensor:
                pq_flange2tool = (self.ft_sensor_mdl.T_mounting2wrench @ self.tool.T_mounting2tip).pose
                tcp_offset_ = pq_flange2tool.xyz + pq_flange2tool.axis_angle
            else:
                tcp_offset_ = self.tool.p_mounting2tip.xyz + self.tool.q_mounting2tip.axis_angle
        elif frame == 'camera':
            pq_flange2cam = self.cam_mdl.T_flange2camera.pose
            tcp_offset_ = pq_flange2cam.xyz + pq_flange2cam.axis_angle
        else:
            raise RuntimeError(f"This code should not be reached. Please check the frame definitions.")

        q_: Sequence[float] = self.rtde.r.getActualQ()
        pose_vec: Sequence[float] = self.rtde.c.getForwardKinematics(q=q_, tcp_offset=tcp_offset_)
        return Pose().from_xyz(pose_vec[:3]).from_axis_angle(pose_vec[3:])

    def get_tcp_offset(self) -> Pose:
        tcp_offset: Sequence[float] = self.rtde.c.getTCPOffset()
        return Pose().from_xyz(tcp_offset[:3]).from_axis_angle(tcp_offset[3:])

    def get_tcp_pose(self) -> Pose:
        tcp_pose: Sequence[float] = self.rtde.r.getActualTCPPose()
        return Pose().from_xyz(tcp_pose[:3]).from_axis_angle(tcp_pose[3:])

    def get_tcp_vel(self) -> Vector6d:
        tcp_vel: Sequence[float] = self.rtde.r.getActualTCPSpeed()
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
        if self.extern_sensor:
            # The default sensor readings are w.r.t the sensor frame
            ft_raw = Vector6d().from_xyzXYZ(self.ft_sensor.FT)
            if frame == 'ft_sensor':
                ft = ft_raw
        else:
            # The default build in sensor readings are w.r.t the arm base frame
            ft = Vector6d()
        return ft

    def get_tcp_force(self, extern: bool = False) -> Vector6d:
        if extern:
            tcp_force = Vector6d().from_xyzXYZ(self.ft_sensor.FT)
        else:
            tcp_force = Vector6d().from_xyzXYZ(self.rtde.r.getActualTCPForce())
        # Compensate Tool mass
        f_tool_wrt_world = self.tool.f_inertia
        f_tool_wrt_ft = (self.q_world2arm * self.get_tcp_pose().q).apply(f_tool_wrt_world, inverse=True)
        t_tool_wrt_ft = Vector3d().from_xyz(np.cross(self.tool.com.xyz, f_tool_wrt_ft.xyz))
        ft_comp = Vector6d().from_Vector3d(f_tool_wrt_ft, t_tool_wrt_ft)
        return tcp_force + ft_comp
