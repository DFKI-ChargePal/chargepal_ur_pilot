from __future__ import annotations

# global
import math
import time
import numpy as np
from pathlib import Path
import spatialmath as sm
from strenum import StrEnum
from time import perf_counter
from contextlib import contextmanager

# local
from ur_pilot import utils
from ur_pilot import config
from ur_pilot.ur_robot import URRobot
from ur_pilot.config_mdl import Config, read_toml
from ur_pilot.control_mode import ControlContextManager
from ur_pilot.end_effector.bota_sensor import BotaFtSensor
from ur_pilot.end_effector.flange_eye_calibration import FlangeEyeCalibration
from ur_pilot.end_effector.models import CameraModel, PlugModelContext, ToolLinkModel, BotaSensONEModel

# typing
from numpy import typing as npt
from camera_kit import CameraBase
from typing import Iterator, Sequence


class EndEffectorFrames(StrEnum):

    FLANGE = 'flange'
    FT_SENSOR = 'ft_sensor'
    TOOL_LINK = 'tool_link'
    PLUG_LIP = 'plug_lip'
    PLUG_TIP = 'plug_tip'
    PLUG_SENSE = 'plug_sense'
    CAMERA = 'camera'


class Pilot:

    R_WORLD2BASE_ = sm.SO3.Rx(np.pi)
    PLUG_NAMES_: list[str] = []
    FT_SENSOR_FRAMES_ = ['world', 'arm_base', 'ft_sensor']
    # EE_FRAMES_ = ['flange', 'ft_sensor', 'tool_link', 'plug_lip', 'plug_tip', 'plug_sense', 'camera']

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
        # Parse configuration
        config_raw = read_toml(self.config_dir.joinpath('ur_pilot.toml'))
        self.cfg = Config(**config_raw)
        # Define robot
        self._robot: URRobot | None = None
        self._context: ControlContextManager | None = None
        self.is_connected = False
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
        # Set up end-effector models
        self.tool_link = ToolLinkModel(**self.cfg.pilot.tool_link.dict())
        self.plug_model = PlugModelContext(self.cfg)
        self.cam: CameraBase | None = None
        self.cam_mdl = CameraModel()

    @property
    def robot(self) -> URRobot:
        if self._robot is not None:
            return self._robot
        else:
            raise RuntimeError("Pilot is not connected to the robot. Please try to connect first.")

    @property
    def context(self) -> ControlContextManager:
        if self._context is not None:
            return self._context
        else:
            raise RuntimeError("Pilot is not connected to the robot. Please try to connect first.")

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
    def rot_world2arm(self) -> sm.SO3:
        return self.R_WORLD2BASE_

    def average_ft_measurement(self, num_meas: int = 100) -> npt.NDArray[np.float_]:
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
        return np.array(np.mean(avg_ft, axis=-1))

    def exit(self) -> None:
        if self._ft_sensor is not None:
            self._ft_sensor.stop()
        self.disconnect()

    def register_ee_cam(self, cam: CameraBase, tf_dir: str | Path = "") -> None:
        self.cam = cam
        if not self.cam.is_calibrated:
            self.cam.load_coefficients()
        T_flange2cam = FlangeEyeCalibration.load_transformation(self.cam, tf_dir)
        self.cam_mdl.T_flange2camera = sm.SE3.CopyFrom(T_flange2cam)

    @contextmanager
    def open_plug_connection(self) -> Iterator[None]:
        self.robot.enable_digital_out(0)
        time.sleep(0.5)
        yield
        self.robot.disable_digital_out(0)
        if self.robot.get_digital_out_state(0):
            raise ValueError(f"Digital output shout be 'LOW' but is 'HIGH'.")

    def __frame_to_offset(self, frame: EndEffectorFrames) -> sm.SE3:
        # if frame not in self.EE_FRAMES_:
        #     raise ValueError(f"Given frame is not available. Please select one of '{self.EE_FRAMES_}'")
        if frame == EndEffectorFrames.FLANGE:
            offset = sm.SE3()
        elif frame == EndEffectorFrames.FT_SENSOR:
            if self.extern_sensor:
                offset = self.ft_sensor_mdl.T_mounting2wrench
            else:
                # Tread internal force torque sensor as mounted in flange frame
                offset = sm.SE3()
        elif frame == EndEffectorFrames.TOOL_LINK:
            if self.extern_sensor:
                offset = self.ft_sensor_mdl.T_mounting2wrench * self.tool_link.T_mounting2link
            else:
                offset = self.tool_link.T_mounting2link
        elif frame == EndEffectorFrames.PLUG_LIP:
            if self.extern_sensor:
                offset = self.ft_sensor_mdl.T_mounting2wrench * self.plug_model.T_mounting2lip
            else:
                offset = self.plug_model.T_mounting2lip
        elif frame == EndEffectorFrames.PLUG_TIP:
            if self.extern_sensor:
                offset = self.ft_sensor_mdl.T_mounting2wrench * self.plug_model.T_mounting2tip
            else:
                offset = self.plug_model.T_mounting2tip
        elif frame == EndEffectorFrames.PLUG_SENSE:
            if self.extern_sensor:
                offset = self.ft_sensor_mdl.T_mounting2wrench * self.plug_model.T_mounting2sense
            else:
                offset = self.plug_model.T_mounting2sense
        elif frame == EndEffectorFrames.CAMERA:
            offset = self.cam_mdl.T_flange2camera
        else:
            raise RuntimeError("This code should not be reached. Please check the frame definitions.")
        return offset

    def get_pose(self, frame: EndEffectorFrames | str = 'flange') -> sm.SE3:
        """ Get pose of the desired frame w.r.t the robot base. This function is independent of the TCP offset defined
            on the robot side.
        Args:
            frame: One of the frame name defined in the class member variable 'EE_FRAMES_'
        Returns:
            6D pose as SE(3) transformation matrix
        """
        frame = EndEffectorFrames(frame)
        tcp_offset = self.__frame_to_offset(frame=frame)
        return self.robot.get_pose(tcp_offset)

    def get_wrench(self, frame: str = 'ft_sensor') -> sm.SpatialForce:
        """ Get force-torque readings w.r.t the desired frame.

        Args:
            frame: One of the frame name defined in the class member variable 'FT_SENSOR_FRAMES_'

        Returns:
            A 6d vector with the sensor readings
        """
        if frame not in self.FT_SENSOR_FRAMES_:
            raise ValueError(f"Given frame is not available. Please select one of '{self.FT_SENSOR_FRAMES_}'")
        # The default build in sensor readings are w.r.t the arm base frame
        if self.extern_sensor:
            ft_raw = self.ft_sensor.FT
        else:
            ft_raw = self.robot.tcp_wrench
        # TODO: Check frame
        # Compensate Tool mass
        f_tool_wrt_world = self.tool_link.f_inertia
        f_tool_wrt_ft: npt.NDArray[np.float_] = self.rot_world2arm * f_tool_wrt_world

        def cross(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float32]:
            # Overwrite numpy cross function to avoid errors by IDE
            a_ = np.array(a, dtype=np.float32)
            b_ = np.array(b, dtype=np.float32)
            return np.cross(a_, b_)

        t_tool_wrt_ft = cross(self.tool_link.com, f_tool_wrt_ft)
        ft_comp = sm.SpatialForce(np.append(f_tool_wrt_ft, t_tool_wrt_ft))
        return ft_raw + ft_comp

    def set_tcp(self, frame: EndEffectorFrames | str = 'tool_link') -> None:
        """ Function to set the tcp relative to the tool flange. """
        frame = EndEffectorFrames(frame)
        offset = self.__frame_to_offset(frame=frame)
        self.robot.set_tcp(offset)

    def move_to_joint_pos(self, q: npt.ArrayLike) -> npt.NDArray[np.float_]:
        self.context.check_mode(expected=self.context.mode_types.POSITION)
        # Move to requested joint position
        self.robot.movej(q)
        new_joint_pos = self.robot.joint_pos
        return new_joint_pos

    def move_to_tcp_pose(self, target: sm.SE3, time_out: float = 3.0) -> tuple[bool, sm.SE3]:
        self.context.check_mode(expected=[self.context.mode_types.POSITION, self.context.mode_types.MOTION])
        fin = False
        # Move to requested TCP pose
        if self.context.mode == self.context.mode_types.POSITION:
            self.robot.movel(target)
            fin = True
        elif self.context.mode == self.context.mode_types.MOTION:
            t_start = perf_counter()
            tgt_3pts = utils.se3_to_3pt_set(target)
            while True:
                # self.robot.motion_controller.update(target_se3)
                self.robot.motion_mode(target)
                cur_3pts = utils.se3_to_3pt_set(self.robot.tcp_pose)
                error = np.mean(np.abs(tgt_3pts, cur_3pts))
                if error <= 0.005:  # 5 mm
                    fin = True
                    break
                elif perf_counter() - t_start > time_out:
                    fin = False
                    break
        return fin, self.robot.tcp_pose

    def push_linear(self, force: npt.ArrayLike, compliant_axes: list[int], duration: float) -> float:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        x_ref = self.robot.tcp_pos
        task_frame = self.robot.tcp_pose
        wrench = np.append(np.reshape(force, 3), np.zeros(3)).tolist()
        t_start = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=wrench)
            time.sleep(self.robot.dt)
            if (perf_counter() - t_start) > duration:
                break
        x_now = self.robot.tcp_pos
        dist: float = np.sum(np.abs(x_now - x_ref))  # L1 norm
        # Stop robot movement.
        self.robot.force_mode(task_frame=task_frame, selection_vector=6 * [0], wrench=6 * [0.0])
        time.sleep(self.robot.dt)
        return dist

    def screw_ee_force_mode(self, torque: float, ang: float, time_out: float) -> bool:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Limit input
        torque = np.clip(torque, 0.0, 5.0)
        wrench_vec = 6 * [0.0]
        compliant_axes = [1, 1, 1, 0, 0, 1]
        # Wrench will be applied with respect to the current TCP pose
        task_frame = self.get_pose('flange')
        # Create target
        ee_jt_pos = self.robot.joint_pos[-1]
        ee_jt_pos_tgt = ee_jt_pos + ang
        # Time observation
        success = False
        t_start = perf_counter()
        # Controller state
        prev_error = np.nan
        i_err = 0.0
        while True:
            # Angular error
            ee_jt_pos_now = self.robot.joint_pos[-1]
            ang_error = (ee_jt_pos_tgt - ee_jt_pos_now)
            p_err = torque * ang_error
            if prev_error is np.NAN:
                d_err = 0.0
            else:
                d_err = 1.0 * (ang_error - prev_error) / self.robot.dt
            i_err = i_err + 3.5e-5 * ang_error / self.robot.dt
            wrench_vec[-1] = np.clip(p_err + d_err + i_err, -torque, torque)
            prev_error = ang_error
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=wrench_vec)
            time.sleep(self.robot.dt)
            t_now = perf_counter()
            if abs(ang_error) < 5e-3:
                success = True
                break
            if t_now - t_start > time_out:
                break
        return success

    def try2_couple_plug(self, T_base2socket: sm.SE3, force: float, time_out: float) -> bool:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Limit input
        force = np.clip(force, 0.0, 50.0)
        wrench_vec = 6 * [0.0]
        selection_vector = [1, 1, 1, 0, 0, 1]

        x_ctrl = utils.PDController(kp=100.0, kd=0.99)
        y_ctrl = utils.PDController(kp=100.0, kd=0.99)
        z_ctrl = utils.PIDController(kp=10.0, kd=0.99, ki=3.5e-5)

        # Get estimation of the plug pose
        T_socket2plug = self.plug_model.T_mounting2lip.inv()
        T_base2plug_est = T_base2socket * T_socket2plug

        # Wrench will be applied with respect to the current TCP pose
        task_frame = T_base2plug_meas = self.get_pose(EndEffectorFrames.TOOL_LINK)
        T_meas2est = T_base2plug_meas.inv() * T_base2plug_est
        p_meas2est_ref = p_meas2est = T_meas2est.t
        success = False
        t_ref = t_start = perf_counter()
        while True:

            wrench_vec[0] = np.clip(x_ctrl.update(p_meas2est[0], self.robot.dt), -50.0, 50.0)
            wrench_vec[1] = np.clip(y_ctrl.update(p_meas2est[1], self.robot.dt), -50.0, 50.0)
            wrench_vec[2] = np.clip(z_ctrl.update(p_meas2est[2], self.robot.dt), -50.0, 50.0)

            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=selection_vector,
                wrench=wrench_vec,
            )

            T_base2plug_meas = self.get_pose(EndEffectorFrames.TOOL_LINK)
            T_meas2est = T_base2plug_meas.inv() * T_base2plug_est
            p_meas2est = T_meas2est.t

            t_now = perf_counter()
            # Check every second if robot is still moving
            if t_now - t_ref > 1.0:
                if np.allclose(p_meas2est_ref, p_meas2est, atol=0.001):
                    fin = True
                    break
                t_ref, p_meas2est_ref = t_now, p_meas2est
            if t_now - t_start > time_out:
                break

        return success

    def try2_decouple_plug(self) -> bool:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        return False

    def try2_unlock_plug(self) -> bool:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        return False

    def try2_lock_plug(self) -> bool:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        return False

    def one_axis_tcp_force_mode(self, axis: str, force: float, time_out: float) -> bool:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        task_frame = self.robot.tcp_pose
        x_ref = self.robot.tcp_pos
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
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=wrench_vec)
            time.sleep(self.robot.dt)
            t_now = perf_counter()
            # Check every second if robot is still moving
            if t_now - t_ref > 1.0:
                x_now = self.robot.tcp_pos
                if np.allclose(x_ref, x_now, atol=0.001):
                    fin = True
                    break
                t_ref, x_ref = t_now, x_now
            if t_now - t_start > time_out:
                break
        # Stop robot movement.
        self.robot.force_mode(task_frame=task_frame, selection_vector=6 * [0], wrench=6 * [0.0])
        time.sleep(self.robot.dt)
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
            mov_dt = self.push_linear(force, compliant_axes, 1.0)
            if mov_dt < 0.001:
                fin = True
                break
            self.relax(0.25)
        return fin

    def plug_in_with_target(self, force: float, T_Base2Socket: sm.SE3, axis: str = 'z', time_out: float = 10.0) -> bool:
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
        task_frame = self.robot.tcp_pose
        wrench_idx = utils.axis2index(axis.lower())
        wrench_vec = 6 * [0.0]
        wrench_vec[wrench_idx] = force
        select_vec = [1, 1, 1, 0, 0, 0]
        select_vec[wrench_idx + 2] = 1
        t_start = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=select_vec,
                wrench=wrench_vec
            )
            time.sleep(self.robot.dt)
            # Get current transformation from base to end-effector
            T_Base2Tip = self.get_pose('tool_tip')
            T_Tip2Socket = T_Base2Tip.inv() * T_Base2Socket
            if T_Tip2Socket.t[wrench_idx] <= -0.032:
                fin = True
                break
            t_now = perf_counter()
            if t_now - t_start > time_out:
                fin = False
                break
        return fin

    def pair_to_socket(self, T_Base2Socket: sm.SE3, force: float = 10.0, time_out: float = 5.0) -> bool:
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
        task_frame = self.robot.tcp_pose
        select_vec = [1, 1, 1, 0, 0, 1]  # Be compliant as possible
        wrench = [0.0, 0.0, abs(force), 0.0, 0.0, 0.0]  # Apply force in tool direction
        # Time observation
        t_start = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=select_vec,
                wrench=wrench
            )
            time.sleep(self.robot.dt)
            # Get current transformation from base to end-effector
            T_Base2Tip = self.get_pose('tool_tip')
            T_Tip2Socket = T_Base2Tip.inv() * T_Base2Socket
            if T_Tip2Socket.tau[2] <= -0.015:
                fin = True
                break
            t_now = perf_counter()
            if t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement.
        self.robot.force_mode(task_frame=task_frame, selection_vector=6 * [0], wrench=6 * [0.0])
        time.sleep(self.robot.dt)
        return fin

    def jog_in_plug(self,
                    T_Base2Socket: sm.SE3, force: float = 20.0, moment: float = 1.0, time_out: float = 5.0) -> bool:
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
        task_frame = self.robot.tcp_pose
        select_vec = [0, 0, 1, 0, 1, 0]
        t_start = perf_counter()
        while True:
            dt = perf_counter() - t_start
            # wrench = [0.0, 0.0, f, m * math.sin(freq * dt), m * math.cos(freq * dt), 0.0]
            wrench = [0.0, 0.0, f, 0.0, m * math.sin(freq * dt), 0.0]
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=select_vec,
                wrench=wrench
            )
            time.sleep(self.robot.dt)
            # Get current transformation from base to end-effector
            T_Base2Tip = self.get_pose('tool_tip')
            T_Tip2Socket = T_Base2Tip.inv() * T_Base2Socket
            if T_Tip2Socket.tau[2] <= -0.032:
                fin = True
                break
            t_now = perf_counter()
            if t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.force_mode(task_frame=task_frame, selection_vector=6 * [0], wrench=6 * [0.0])
        time.sleep(self.robot.dt)
        return fin

    def tcp_force_mode(self,
                       wrench: npt.ArrayLike, compliant_axes: list[int], distance: float, time_out: float) -> bool:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        task_frame = self.robot.tcp_pose
        t_start = perf_counter()
        x_ref = self.robot.tcp_pos
        ft_vec = np.reshape(wrench, 6).tolist()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=ft_vec)
            time.sleep(self.robot.dt)
            x_now = self.robot.tcp_pos
            l2_norm_dist = np.linalg.norm(x_now - x_ref)
            t_now = perf_counter()
            if l2_norm_dist >= distance:
                fin = True
                break
            elif t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.force_mode(task_frame=task_frame, selection_vector=6 * [0], wrench=6 * [0.0])
        time.sleep(self.robot.dt)
        return fin

    def frame_force_mode(self,
                         wrench: npt.ArrayLike, 
                         compliant_axes: list[int], 
                         distance: float, 
                         time_out: float, 
                         frame: str) -> bool:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        task_frame = self.get_pose(frame=frame)
        t_start = perf_counter()
        x_ref = self.robot.tcp_pos
        ft_vec = np.reshape(wrench, 6).tolist()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=ft_vec)
            time.sleep(self.robot.dt)
            x_now = self.robot.tcp_pos
            l2_norm_dist = np.linalg.norm(x_now - x_ref)
            t_now = perf_counter()
            if l2_norm_dist >= distance:
                fin = True
                break
            elif t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.force_mode(task_frame=task_frame, selection_vector=6 * [0], wrench=6 * [0.0])
        time.sleep(self.robot.dt)
        return fin

    def find_contact_point(self, direction: Sequence[int], time_out: float) -> tuple[bool, sm.SE3]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Map direction to wrench
        wrench = np.clip([10.0 * d for d in direction], -10.0, 10.0).tolist()
        selection_vector = [1 if d != 0 else 0 for d in direction]
        task_frame = sm.SE3()  # Robot base
        x_ref = self.robot.tcp_pos
        t_start = t_ref = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=selection_vector,
                wrench=wrench
            )
            time.sleep(self.robot.dt)
            t_now = perf_counter()
            # Check every 500 milliseconds if robot is still moving
            if t_now - t_ref > 0.5:
                x_now = self.robot.tcp_pos
                if np.allclose(x_ref, x_now, atol=0.001):
                    fin = True
                    break
                t_ref, x_ref = t_now, x_now
            elif t_now - t_start > time_out:
                fin = False
                break
        return fin, self.get_pose(frame='flange')

    def sensing_depth(self, T_Base2Target: sm.SE3, time_out: float) -> tuple[bool, sm.SE3]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Parameter set. Sensing is in tool direction
        selection_vector = [0, 0, 1, 0, 0, 0]
        wrench = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        task_frame = self.robot.tcp_pose
        # Process observation variables
        x_ref = self.robot.tcp_pos
        t_start = t_ref = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=selection_vector,
                wrench=wrench
            )
            time.sleep(self.robot.dt)
            t_now = perf_counter()
            # Check every 500 millisecond if robot is still moving
            if t_now - t_ref > 0.5:
                x_now = self.robot.tcp_pos
                if np.allclose(x_ref, x_now, atol=0.001):
                    fin = True
                    break
                t_ref, x_ref = t_now, x_now
            elif t_now - t_start > time_out:
                fin = False
                break
        if not fin:
            return fin, sm.SE3()
        else:
            T_Base2Target_meas = sm.SE3.Rt(R=T_Base2Target.R, t=self.robot.tcp_pos)
            return fin, T_Base2Target_meas

    def plug_in_motion_mode(self, target: sm.SE3, time_out: float) -> tuple[bool, sm.SE3]:
        self.context.check_mode(expected=self.context.mode_types.MOTION)
        t_start = perf_counter()
        while True:
            # Move linear to target
            self.robot.motion_mode(target)
            # Check error in plugging direction
            act_pose = self.robot.tcp_pose
            error_p = target.t - act_pose.t
            # Rotate in tcp frame
            error_p_tcp = act_pose.inv() * error_p
            if abs(error_p_tcp.t[-1]) <= 0.005:
                fin = True
                break
            elif perf_counter() - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.force_mode(sm.SE3(), 6 * [0], 6 * [0.0])
        time.sleep(self.robot.dt)
        return fin, self.robot.tcp_pose

    def relax(self, time_duration: float) -> sm.SE3:
        self.context.check_mode(expected=[
            self.context.mode_types.FORCE, self.context.mode_types.MOTION, self.context.mode_types.HYBRID
        ])
        task_frame = self.robot.tcp_pose
        t_start = perf_counter()
        # Apply zero wrench and be compliant in all axes
        wrench = 6 * [0.0]
        compliant_axes = 6 * [1]
        while perf_counter() - t_start < time_duration:
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=wrench)
            time.sleep(self.robot.dt)
        # Stop robot movement.
        self.robot.force_mode(task_frame=task_frame, selection_vector=6 * [0], wrench=6 * [0.0])
        time.sleep(self.robot.dt)
        return self.robot.tcp_pose

    def retreat(self,
                task_frame: sm.SE3,
                direction: Sequence[int],
                distance: float = 0.02,
                time_out: float = 3.0) -> tuple[bool, sm.SE3]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Map direction to wrench
        wrench = np.clip([10.0 * d for d in direction], -10.0, 10.0).tolist()
        selection_vector = [1 if d != 0 else 0 for d in direction]
        t_start = perf_counter()
        x_ref = self.robot.tcp_pos
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=selection_vector,
                wrench=wrench
            )
            time.sleep(self.robot.dt)
            t_now = perf_counter()
            x_now = self.robot.tcp_pos
            l2_norm_dist = np.linalg.norm(x_now - x_ref)
            if l2_norm_dist >= distance:
                fin = True
                break
            elif t_now - t_start > time_out:
                fin = False
                break
        return fin, self.robot.tcp_pose

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

    def connect(self) -> None:
        if not self.is_connected:
            # Set up ur_control
            self._robot = URRobot(self.config_dir.joinpath('ur_control.toml'), self.cfg)
            self._context = ControlContextManager(self.robot, self.cfg)
            self.set_tcp()
            self.is_connected = True
            # Set control context manager

    def disconnect(self) -> None:
        """ Exit function which will be called from the context manager at the end """
        if self.is_connected:
            self.context.exit()
            self.robot.disconnect()
            self._robot = None
            self._context = None
            self.is_connected = False
