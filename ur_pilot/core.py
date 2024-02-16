from __future__ import annotations

# global
import math
import time
import numpy as np
from pathlib import Path
import spatialmath as sm
from enum import auto, Enum
from time import perf_counter
from contextlib import contextmanager

from ur_control.utils import ur_format2tr

from rigmopy import utils_math as rp_math
from rigmopy import Pose, Quaternion, Transformation, Vector3d, Vector6d

# local
from ur_pilot import utils
from ur_pilot import config
from ur_pilot.robot import Robot
from ur_pilot.control_mode import ControlContextManager
from ur_pilot.config_mdl import Config, read_toml
from ur_pilot.utils import SpatialPDController
from ur_pilot.end_effector.bota_sensor import BotaFtSensor
from ur_pilot.end_effector.flange_eye_calibration import FlangeEyeCalibration
from ur_pilot.end_effector.models import CameraModel, ToolModel, BotaSensONEModel


# typing
from numpy import typing as npt
from camera_kit import CameraBase
from typing import Any, Iterator, Sequence


@contextmanager
def connect(config_dir: Path | None = None) -> Iterator[Pilot]:
    pilot = Pilot(config_dir)
    try:
        yield pilot
    finally:
        pilot.disconnect()


class ArmState(Enum):
    NA = auto()  # not applicable
    HOME = auto()
    DRIVE = auto()
    CCS_SIDE = auto()
    TYPE2_SIDE = auto()

    @staticmethod
    def from_str(name: str) -> ArmState:
        enum_state = None
        for es in ArmState:
            if name.upper() == es.name:
                enum_state = es
        if enum_state is None:
            raise KeyError(f"Can't match state name '{name.upper()}' with any enum name. "
                           f"Possible values are: {[e.name for e in ArmState]}")
        return enum_state


class JointState:

    available_states = {
        'na': np.zeros(6, dtype=np.float32),
        'home': np.ones(6, dtype=np.float32),
    }

    def __init__(self, robot: Robot):
        self.state = 'na'
        self.value = np.zeros(6, np.float32)
        self.robot = robot
        self.check_state()

    def check_state(self) -> None:
        if self.state not in self.available_states.keys():
            raise ValueError(f"Trying to bring arm in an unknown state: {self.state}. "
                             f"Known states are: {self.available_states}")
        if self.state != 'na':
            # TODO: Check against real arm values
            print(f"Check if arm is close to state values for state {self.state}!")

    def check_existence(self, name: str) -> bool:
        exs = False
        # Only work with lowercase strings
        name = name.lower()
        if name in self.available_states.keys():
            exs = True
        return exs

    def add(self, name: str, value: npt.ArrayLike) -> None:
        # Only work with lowercase strings
        name = name.lower()
        if name in self.available_states.keys():
            raise RuntimeError(f"State with name '{name}' already exists.")
        val = np.array(np.reshape(value, 6), dtype=np.float32)
        self.available_states[name] = val

    def get_state(self) -> tuple[str, npt.NDArray[np.float32]]:
        self.check_state()
        return self.state, self.value

    def set_state(self, name: str) -> None:
        # Only work with lowercase strings
        name = name.lower()
        if name not in self.available_states.keys():
            raise RuntimeError(f"Can't find a state with name '{name}'")
        else:
            self.state = name
            self.value = self.available_states[name]
        self.check_state()


class Maneuvers:

    def __init__(self, robot: Robot, joint_state: JointState):
        self.maneuvers: dict[str, dict[str, Any]] = {}
        self.joint_state = joint_state
        self.robot = robot

    def add(self, name: str, start: str, stop: str, waypoints: list[npt.ArrayLike] | None = None) -> None:
        # Only work with lowercase strings
        name = name.lower()
        if name in self.maneuvers.keys():
            raise RuntimeError(f"Maneuver with name '{name}' already exist.")
        if not self.joint_state.check_existence(start):
            raise ValueError(f"Start state with name '{start}' doesn't exist!")
        if not self.joint_state.check_existence(stop):
            raise ValueError(f"Stop state with name '{stop}' doesn't exist!")
        self.maneuvers[name] = {
            'start': start,
            'stop': stop,
            'waypoints': waypoints
        }

    def execute(self, name: str) -> None:
        # Only work with lowercase strings
        name = name.lower()
        maneuver = self.maneuvers[name]
        if self.joint_state != maneuver['start']:
            raise RuntimeError(f"Arm not in start state. Can't execute maneuver.")
        if maneuver['waypoints'] is None:
            pass
        else:
            pass


class Pilot:

    Q_WORLD2BASE_ = Quaternion().from_euler_angle([np.pi, 0.0, 0.0])
    FT_SENSOR_FRAMES_ = ['world', 'arm_base', 'ft_sensor']
    EE_FRAMES_ = ['flange', 'ft_sensor', 'tool_tip', 'tool_sense', 'camera']

    def __init__(self, config_dir: Path | None = None) -> None:
        """ Core class to interact with the robot

        Args:
            config_dir: Path to a configuration folder

        Raises:
            FileNotFoundError: Check if configuration file exists
        """
        if config_dir is not None and not config_dir.is_dir():
            raise NotADirectoryError(f"Can't find directory with path: {config_dir}")
        if config_dir is None:
            self.config_dir = Path(config.__file__).parent
        else:
            self.config_dir = config_dir
        # Set up ur_control
        self.robot = Robot(self.config_dir.joinpath('ur_control.toml'))

        config_raw = read_toml(self.config_dir.joinpath('ur_pilot.toml'))
        self.cfg = Config(**config_raw)
        self.context = ControlContextManager(self.robot, self.cfg)

        # Set up end-effector
        if self.cfg.pilot.ft_sensor is None:
            self.extern_sensor = False
            self._ft_sensor = None
            self._ft_sensor_mdl = None
        else:
            self.extern_sensor = True
            self._ft_sensor = BotaFtSensor(**self.cfg.pilot.ft_sensor.dict())
            self._ft_sensor_mdl = BotaSensONEModel()
            self._ft_sensor.start()

        self._motion_pd: SpatialPDController | None = None
        self._force_pd: SpatialPDController | None = None

        self.tool = ToolModel(**self.cfg.pilot.tool_model.dict())
        self.set_tcp()
        self.cam: CameraBase | None = None
        self.cam_mdl = CameraModel()

        # Constants
        self.dt = 1 / self.robot.cfg.robot.rtde_freq
        self.error_scale_motion_mode = 1.0
        self.force_limit = 0.0

    # @property
    # def home_joint_config(self) -> tuple[float, ...]:
    #     if self.cfg.pilot.home_radians is not None:
    #         return tuple(self.cfg.pilot.home_radians)
    #     else:
    #         raise ValueError("Home joint configuration was not set.")
    #
    # def overwrite_home_joint_config(self, joint_pos: npt.ArrayLike) -> None:
    #     jp = np.reshape(joint_pos, 6).tolist()
    #     config_raw = read_toml(self.config_fp)
    #     config_raw['robot']['home_radians'] = jp
    #     write_toml(config_raw, self.config_fp)

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
        self.disconnect()

    def register_ee_cam(self, cam: CameraBase, tf_dir: str = "") -> None:
        self.cam = cam
        if not self.cam.is_calibrated:
            self.cam.load_coefficients()
        T_flange2cam = FlangeEyeCalibration.load_transformation(self.cam, tf_dir)
        self.cam_mdl.T_flange2camera = Transformation().from_trans_matrix(T_flange2cam)

    @contextmanager
    def open_plug_connection(self) -> Iterator[None]:
        self.robot.enable_digital_out(0)
        time.sleep(0.5)
        yield
        self.robot.disable_digital_out(0)
        if self.robot.get_digital_out_state(0):
            raise ValueError(f"Digital output shout be 'LOW' but is 'HIGH'.")

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
                # LOGGER.warning(f"External ft_sensor is not configured. Return pose of the flange frame.")
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
            raise RuntimeError("This code should not be reached. Please check the frame definitions.")
        return offset

    def get_pose(self, frame: str = 'flange') -> Pose:
        """ Get pose of the desired frame w.r.t the robot base. This function is independent of the TCP offset defined
            on the robot side.
        Args:
            frame: One of the frame name defined in the class member variable 'EE_FRAMES_'
        Returns:
            6D pose
        """
        tcp_offset = self.__frame_to_offset(frame=frame)
        q = self.robot.joint_pos
        pose_vec: Sequence[float] = self.robot.rtde_controller.getForwardKinematics(q=q, tcp_offset=tcp_offset)
        return Pose().from_xyz(pose_vec[:3]).from_axis_angle(pose_vec[3:])

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
            tcp_force = Vector6d().from_xyzXYZ(self.robot.tcp_force)
        # Compensate Tool mass
        f_tool_wrt_world = self.tool.f_inertia
        f_tool_wrt_ft = (self.q_world2arm * Quaternion().from_matrix(
            self.robot.tcp_pose.R)).apply(f_tool_wrt_world, inverse=True)
        t_tool_wrt_ft = Vector3d().from_xyz(np.cross(self.tool.com.xyz, f_tool_wrt_ft.xyz))
        ft_comp = Vector6d().from_Vector3d(f_tool_wrt_ft, t_tool_wrt_ft)
        return tcp_force + ft_comp

    def set_tcp(self, frame: str = 'tool_tip') -> None:
        """ Function to set the tcp relative to the tool flange. """
        offset = ur_format2tr(self.__frame_to_offset(frame=frame))
        self.robot.set_tcp(offset)

    def set_up_force_mode(self, gain: float | None = None, damping: float | None = None) -> None:
        gain_scaling = gain if gain else self.cfg.pilot.force_mode.gain
        damping_fact = damping if damping else self.cfg.pilot.force_mode.damping
        self.robot.rtde_controller.zeroFtSensor()
        self.robot.rtde_controller.forceModeSetGainScaling(gain_scaling)
        self.robot.rtde_controller.forceModeSetDamping(damping_fact)

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
        self.error_scale_motion_mode = error_scale if error_scale else self.cfg.pilot.motion_mode.error_scale
        self.force_limit = force_limit if force_limit else self.cfg.pilot.motion_mode.force_limit
        Kp = Kp_6 if Kp_6 is not None else self.cfg.pilot.motion_mode.Kp
        Kd = Kd_6 if Kd_6 is not None else self.cfg.pilot.motion_mode.Kd
        self._motion_pd = SpatialPDController(Kp_6=Kp, Kd_6=Kd)
        self.set_up_force_mode(gain=ft_gain, damping=ft_damping)

    def motion_mode(self, target: Pose) -> None:
        """ Function to update motion target and let the motion controller keep running

        Args:
            target: Target pose of the TCP
        """
        task_frame = 6 * (0.0, )  # Move w.r.t. robot base
        # Get current Pose
        actual = self.robot.get_tcp_pose()
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
        self.robot.force_mode(
            task_frame=ur_format2tr(task_frame),
            selection_vector=6 * (1,),
            wrench=f_net_clip.tolist(),
            )
        # pose = sm.SE3.Rt(R=sm.UnitQuaternion(target.q.wxyz).SO3(), t=target.p.xyz)
        # self.motion_mode_ur(pose)

    def stop_motion_mode(self) -> None:
        """ Function to set robot back in normal position control mode. """
        self.motion_pd.reset()
        self.robot.stop_force_mode()

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
        self.error_scale_motion_mode = error_scale if error_scale else self.cfg.pilot.hybrid_mode.error_scale
        self.force_limit = force_limit if force_limit else self.cfg.pilot.hybrid_mode.force_limit
        f_Kp = Kp_6_force if Kp_6_force is not None else self.cfg.pilot.hybrid_mode.Kp_force
        f_Kd = Kd_6_force if Kd_6_force is not None else self.cfg.pilot.hybrid_mode.Kd_force
        m_Kp = Kp_6_motion if Kp_6_motion is not None else self.cfg.pilot.hybrid_mode.Kp_motion
        m_Kd = Kd_6_motion if Kd_6_motion is not None else self.cfg.pilot.hybrid_mode.Kd_motion
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
        cur_pose = self.robot.get_tcp_pose()
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
        self.robot.force_mode(task_frame=ur_format2tr(task_frame), selection_vector=6 * (1,), wrench=f_net_clip.tolist())

    def stop_hybrid_mode(self) -> None:
        """ Function to set robot back in default control mode """
        self.force_pd.reset()
        self.motion_pd.reset()
        self.robot.stop_force_mode()

    def move_home(self) -> list[float]:
        self.context.check_mode(expected=self.context.mode_types.POSITION)
        self.robot.move_home()
        # self.robot.move_home()
        new_j_pos = self.robot.get_joint_pos()
        return new_j_pos

    def move_to(self, goal_state: str) -> npt.NDArray[np.float_]:
        goal_ = ArmState.from_str(goal_state)
        if goal_ == ArmState.HOME:
            self.move_home()
            self.arm_state = ArmState.HOME
        elif goal_ == ArmState.DRIVE:
            self.robot.movej([])
            self.arm_state = ArmState.DRIVE
        elif goal_ == ArmState.CCS_SIDE:
            self.robot.movej([])
            self.arm_state = ArmState.CCS_SIDE
        elif goal_ == ArmState.TYPE2_SIDE:
            self.robot.movej([])
            self.arm_state = ArmState.TYPE2_SIDE
        return self.robot.joint_pos

    def move_to_joint_pos(self, q: Sequence[float]) -> list[float]:
        self.context.check_mode(expected=self.context.mode_types.POSITION)
        # Move to requested joint position
        self.robot.movej(q)
        new_joint_pos = self.robot.get_joint_pos()
        return new_joint_pos

    def move_to_tcp_pose(self, target: Pose, time_out: float = 3.0) -> tuple[bool, Pose]:
        self.context.check_mode(expected=[self.context.mode_types.POSITION, self.context.mode_types.MOTION])
        fin = False
        r = sm.UnitQuaternion(target.q.wxyz).SO3()
        target_se3 = sm.SE3.Rt(R=r, t=target.p.xyz)
        # Move to requested TCP pose
        if self.context.mode == self.context.mode_types.POSITION:
            # self.robot.move_l(target)
            self.robot.movel(target_se3)
            fin = True
        elif self.context.mode == self.context.mode_types.MOTION:
            t_start = perf_counter()
            tgt_3pts = target.to_3pt_set()
            while True:
                # self.robot.motion_controller.update(target_se3)
                self.motion_mode(target)
                cur_3pts = self.robot.get_tcp_pose().to_3pt_set()
                error = np.mean(np.abs(tgt_3pts, cur_3pts))
                if error <= 0.005:  # 5 mm
                    fin = True
                    break
                elif perf_counter() - t_start > time_out:
                    fin = False
                    break
        new_pose = self.robot.get_tcp_pose()
        return fin, new_pose

    def push_linear(self, force: Vector3d, compliant_axes: list[int], duration: float) -> float:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        wrench = Vector6d().from_Vector3d(force, Vector3d()).xyzXYZ
        x_ref = np.array(X_tcp.xyz, dtype=np.float32)
        t_start = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=ur_format2tr(task_frame),
                selection_vector=compliant_axes,
                wrench=wrench)
            if (perf_counter() - t_start) > duration:
                break
        x_now = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
        dist: float = np.sum(np.abs(x_now - x_ref))  # L1 norm
        # Stop robot movement.
        self.robot.force_mode(task_frame=ur_format2tr(task_frame), selection_vector=6 * [0], wrench=6 * [0.0])
        return dist

    def screw_ee_force_mode(self, torque: float, ang: float, time_out: float) -> bool:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Limit input
        torque = np.clip(torque, 0.0, 5.0)
        wrench_vec = 6 * [0.0]
        compliant_axes = [1, 1, 1, 0, 0, 1]
        # Wrench will be applied with respect to the current TCP pose
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        # Create target
        ee_jt_pos = self.robot.get_joint_pos()[-1]
        ee_jt_pos_tgt = ee_jt_pos + ang
        # Time observation
        success = False
        t_start = perf_counter()
        # Controller state
        prev_error = np.nan
        i_err = 0.0
        while True:
            # Angular error
            ee_jt_pos_now = self.robot.get_joint_pos()[-1]
            ang_error = (ee_jt_pos_tgt - ee_jt_pos_now)
            p_err = torque * ang_error
            if prev_error is np.NAN:
                d_err = 0.0
            else:
                d_err = 1.0 * (ang_error - prev_error) / self.dt
            i_err = i_err + 3.5e-5 * ang_error / self.dt
            wrench_vec[-1] = np.clip(p_err + d_err + i_err, -torque, torque)
            prev_error = ang_error
            # Apply wrench
            self.robot.force_mode(
                task_frame=ur_format2tr(task_frame),
                selection_vector=compliant_axes,
                wrench=wrench_vec)
            t_now = perf_counter()
            if abs(ang_error) < 5e-3:
                success = True
                break
            if t_now - t_start > time_out:
                break
        return success

    def one_axis_tcp_force_mode(self, axis: str, force: float, time_out: float) -> bool:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        x_ref = np.array(X_tcp.xyz, dtype=np.float32)
        # Check if axis is valid
        if not axis.lower() in ['x', 'y', 'z']:
            raise ValueError(f"Only linear axes allowed.")
        wrench_idx = utils.axis2index(axis.lower())
        wrench_vec = 6 * [0.0]
        wrench_vec[wrench_idx] = force
        compliant_axes = [1, 1, 1, 0, 0, 0]
        compliant_axes[wrench_idx + 2] = 1
        # Time observation
        fin = False
        t_ref = t_start = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=ur_format2tr(task_frame),
                selection_vector=compliant_axes,
                wrench=wrench_vec)
            t_now = perf_counter()
            # Check every second if robot is still moving
            if t_now - t_ref > 1.0:
                x_now = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
                if np.allclose(x_ref, x_now, atol=0.001):
                    fin = True
                    break
                t_ref, x_ref = t_now, x_now
            if t_now - t_start > time_out:
                break
        # Stop robot movement.
        self.robot.force_mode(task_frame=ur_format2tr(task_frame), selection_vector=6 * [0], wrench=6 * [0.0])
        return fin

    def plug_in_force_ramp(
            self, f_axis: str = 'z', f_start: float = 0, f_end: float = 50, duration: float = 10) -> bool:
        """ Method to plug in with gradual increasing force

        Args:
            f_axis:   Plugging direction
            f_start:  Start force that will be applied at the beginning
            f_end:    Maximum force that will be applied
            duration: Time period for the process. Has to be at least one second

        Returns:
            True when there is no more movement; False otherwise
        """
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        fin = False
        wrench_idx = utils.axis2index(f_axis.lower())
        compliant_axes = [1, 1, 1, 0, 0, 0]
        compliant_axes[wrench_idx + 2] = 1
        force_ramp = utils.ramp(f_start, f_end, duration)
        force = 3 * [0.0]
        for f in force_ramp:
            force[wrench_idx] = f
            mov_dt = self.push_linear(Vector3d().from_xyz(force), compliant_axes, 1.0)
            if mov_dt < 0.0025:
                fin = True
                break
            self.relax(0.25)
        return fin

    def plug_in_with_target(
            self, force: float, T_Base2Socket: Transformation, axis: str = 'z',  time_out: float = 10.0) -> bool:
        """

        Args:
            force:         Plugging force
            T_Base2Socket: Target in the arm base frame
            axis:          Plugging direction
            time_out:      Maximum time period

        Returns:
            Success
        """
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        wrench_idx = utils.axis2index(axis.lower())
        wrench_vec = 6 * [0.0]
        wrench_vec[wrench_idx] = force
        select_vec = [1, 1, 1, 0, 0, 0]
        select_vec[wrench_idx + 2] = 1
        t_start = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=ur_format2tr(task_frame),
                selection_vector=select_vec,
                wrench=wrench_vec
            )
            # Get current transformation from base to end-effector
            T_Base2Tip = Transformation().from_pose(self.get_pose('tool_tip'))
            T_Tip2Socket = T_Base2Tip.inverse() @ T_Base2Socket
            if T_Tip2Socket.tau[wrench_idx] <= -0.032:
                fin = True
                break
            t_now = perf_counter()
            if t_now - t_start > time_out:
                fin = False
                break
        return fin

    def pair_to_socket(self, T_Base2Socket: Transformation, force: float = 10.0, time_out: float = 5.0) -> bool:
        """ Pair the plug to the socket while using low force to insert for 1.5cm

        Args:
            T_Base2Socket: Transformation from robot base to socket (target).
            force: Used force. Should be low to avoid damage.
            time_out: Time window to execute this action

        Returns:
            Success flag
        """
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        select_vec = [1, 1, 1, 0, 0, 1]  # Be compliant as possible
        wrench = [0.0, 0.0, abs(force), 0.0, 0.0, 0.0]  # Apply force in tool direction
        # Time observation
        t_start = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=ur_format2tr(task_frame),
                selection_vector=select_vec,
                wrench=wrench
            )
            # Get current transformation from base to end-effector
            T_Base2Tip = Transformation().from_pose(self.get_pose('tool_tip'))
            T_Tip2Socket = T_Base2Tip.inverse() @ T_Base2Socket
            if T_Tip2Socket.tau[2] <= -0.015:
                fin = True
                break
            t_now = perf_counter()
            if t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement.
        self.robot.force_mode(task_frame=ur_format2tr(task_frame), selection_vector=6 * [0], wrench=6 * [0.0])
        return fin
    
    def jog_in_plug(self,
                    T_Base2Socket: Transformation, force: float = 20.0, moment: float = 1.0,
                    time_out: float = 5.0) -> bool:
        """ Push in plug with additional (sinusoidal) jiggling

        Args:
            T_Base2Socket: Transformation from robot base to socket (target).
            force:
            moment:
            time_out: Time window to execute this action

        Returns:
            Success flag
        """
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # wrench parameter
        freq = 10.0
        f = abs(force)
        m = abs(moment)
        # Wrench will be applied with respect to the current TCP pose
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        select_vec = [0, 0, 1, 0, 1, 0]
        t_start = perf_counter()
        while True:
            dt = perf_counter() - t_start
            # wrench = [0.0, 0.0, f, m * math.sin(freq * dt), m * math.cos(freq * dt), 0.0]
            wrench = [0.0, 0.0, f, 0.0, m * math.sin(freq * dt), 0.0]
            # Apply wrench
            self.robot.force_mode(
                task_frame=ur_format2tr(task_frame),
                selection_vector=select_vec,
                wrench=wrench
            )
            # Get current transformation from base to end-effector
            T_Base2Tip = Transformation().from_pose(self.get_pose('tool_tip'))
            T_Tip2Socket = T_Base2Tip.inverse() @ T_Base2Socket
            if T_Tip2Socket.tau[2] <= -0.032:
                fin = True
                break
            t_now = perf_counter()
            if t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.force_mode(task_frame=ur_format2tr(task_frame), selection_vector=6 * [0], wrench=6 * [0.0])
        return fin

    def tcp_force_mode(self,
                       wrench: Vector6d,
                       compliant_axes: list[int],
                       distance: float,
                       time_out: float) -> bool:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        t_start = perf_counter()
        x_ref = np.array(X_tcp.xyz, dtype=np.float32)
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=ur_format2tr(task_frame),
                selection_vector=compliant_axes,
                wrench=wrench.xyzXYZ)
            x_now = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
            l2_norm_dist = np.linalg.norm(x_now - x_ref)
            t_now = perf_counter()
            if l2_norm_dist >= distance:
                fin = True
                break
            elif t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.force_mode(task_frame=ur_format2tr(task_frame), selection_vector=6 * [0], wrench=6 * [0.0])
        return fin

    def find_contact_point(self, direction: Sequence[int], time_out: float) -> tuple[bool, Pose]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Map direction to wrench
        wrench = np.clip([10.0 * d for d in direction], -10.0, 10.0).tolist()
        selection_vector = [1 if d != 0 else 0 for d in direction]
        task_frame = 6 * [0.0]  # Robot base
        x_ref = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
        t_start = t_ref = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=ur_format2tr(task_frame),
                selection_vector=selection_vector,
                wrench=wrench
                )
            t_now = perf_counter()
            # Check every 500 milliseconds if robot is still moving
            if t_now - t_ref > 0.5:
                x_now = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
                if np.allclose(x_ref, x_now, atol=0.001):
                    fin = True
                    break
                t_ref, x_ref = t_now, x_now
            elif t_now - t_start > time_out:
                fin = False
                break
        return fin, self.get_pose(frame='flange')

    def sensing_depth(self, T_Base2Target: Transformation, time_out: float) -> tuple[bool, Transformation]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Parameter set. Sensing is in tool direction
        selection_vector = [0, 0, 1, 0, 0, 0]
        wrench = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        # Process observation variables
        x_ref = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
        t_start = t_ref = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=ur_format2tr(task_frame),
                selection_vector=selection_vector,
                wrench=wrench
                )
            t_now = perf_counter()
            # Check every 500 millisecond if robot is still moving
            if t_now - t_ref > 0.5:
                x_now = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
                if np.allclose(x_ref, x_now, atol=0.001):
                    fin = True
                    break
                t_ref, x_ref = t_now, x_now
            elif t_now - t_start > time_out:
                fin = False
                break
        if not fin:
            return fin, Transformation()
        else:
            tau_Base2Target = self.robot.get_tcp_pose().xyz
            rot = T_Base2Target.rot_matrix
            tau = np.array(tau_Base2Target)
            return fin, Transformation().from_rot_tau(rot_mat=rot, tau=tau)

    def plug_in_motion_mode(self, target: Pose, time_out: float) -> tuple[bool, Pose]:
        self.context.check_mode(expected=self.context.mode_types.MOTION)
        t_start = perf_counter()
        while True:
            # Move linear to target
            self.motion_mode(target)
            # Check error in plugging direction
            act_pose = self.robot.get_tcp_pose()
            error_p = target.p - act_pose.p
            # Rotate in tcp frame
            error_p_tcp = act_pose.q.apply(error_p, inverse=True)
            if abs(error_p_tcp.xyz[-1]) <= 0.005:
                fin = True
                break
            elif perf_counter() - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.force_mode(ur_format2tr(6 * [0.0]), 6 * [0], 6 * [0.0])
        return fin, self.robot.get_tcp_pose()

    def relax(self, time_duration: float) -> Pose:
        self.context.check_mode(expected=[
            self.context.mode_types.FORCE, self.context.mode_types.MOTION, self.context.mode_types.HYBRID
            ])
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        t_start = perf_counter()
        # Apply zero wrench and be compliant in all axes
        wrench = 6 * [0.0]
        compliant_axes = 6 * [1]
        while perf_counter() - t_start < time_duration:
            self.robot.force_mode(
                task_frame=ur_format2tr(task_frame),
                selection_vector=compliant_axes,
                wrench=wrench)
        # Stop robot movement.
        self.robot.force_mode(task_frame=ur_format2tr(task_frame), selection_vector=6 * [0], wrench=6 * [0.0])
        return self.robot.get_tcp_pose()

    def retreat(self,
                task_frame: Sequence[float], direction: Sequence[int],
                distance: float = 0.02, time_out: float = 3.0) -> tuple[bool, Pose]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Map direction to wrench
        wrench = np.clip([10.0 * d for d in direction], -10.0, 10.0).tolist()
        selection_vector = [1 if d != 0 else 0 for d in direction]
        t_start = perf_counter()
        x_ref = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=ur_format2tr(task_frame),
                selection_vector=selection_vector,
                wrench=wrench
                )
            t_now = perf_counter()
            x_now = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
            l2_norm_dist = np.linalg.norm(x_now - x_ref)
            if l2_norm_dist >= distance:
                fin = True
                break
            elif t_now - t_start > time_out:
                fin = False
                break
        return fin, self.robot.get_tcp_pose()

    def move_joints_random(self) -> npt.NDArray[np.float32]:
        self.context.check_mode(expected=self.context.mode_types.POSITION)
        # Move to home joint position
        self.robot.move_home()
        # Move to random joint positions near to the home configuration
        home_q = np.array(self.robot.cfg.robot.home_radians, dtype=np.float32)
        tgt_joint_q = np.array(home_q + (np.random.rand(6) * 2.0 - 1.0) * 0.075, dtype=np.float32)
        self.robot.movej(tgt_joint_q)
        # Move back to home joint positions
        self.robot.move_home()
        return tgt_joint_q

    def disconnect(self) -> None:
        """ Exit function which will be called from the context manager at the end """
        self.context.exit()
        self.robot.disconnect()
