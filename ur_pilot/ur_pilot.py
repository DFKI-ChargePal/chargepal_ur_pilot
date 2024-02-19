from __future__ import annotations

# global
import math
import time
import numpy as np
from pathlib import Path
import spatialmath as sm
from time import perf_counter
from contextlib import contextmanager
from ur_control.utils import ur_format2tr
from rigmopy import Pose, Quaternion, Transformation, Vector3d, Vector6d

# local
from ur_pilot import utils
from ur_pilot import config
from ur_pilot.ur_robot import URRobot
from ur_pilot.config_mdl import Config, read_toml
from ur_pilot.control_mode import ControlContextManager
from ur_pilot.end_effector.bota_sensor import BotaFtSensor
from ur_pilot.end_effector.flange_eye_calibration import FlangeEyeCalibration
from ur_pilot.end_effector.models import CameraModel, ToolModel, BotaSensONEModel

# typing
from numpy import typing as npt
from camera_kit import CameraBase
from typing import Iterator, Sequence


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
        # Parse configuration
        config_raw = read_toml(self.config_dir.joinpath('ur_pilot.toml'))
        self.cfg = Config(**config_raw)
        # Set up ur_control
        self.robot = URRobot(self.config_dir.joinpath('ur_control.toml'), self.cfg)
        # Set control context manager
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

        self.tool = ToolModel(**self.cfg.pilot.tool_model.dict())
        self.set_tcp()
        self.cam: CameraBase | None = None
        self.cam_mdl = CameraModel()

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
        self.cam_mdl.T_flange2camera = sm.SE3.CopyFrom(T_flange2cam)

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

    def move_home(self) -> list[float]:
        self.context.check_mode(expected=self.context.mode_types.POSITION)
        self.robot.move_home()
        # self.robot.move_home()
        new_j_pos = self.robot.get_joint_pos()
        return new_j_pos

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
                self.robot.motion_mode(target)
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
                d_err = 1.0 * (ang_error - prev_error) / self.robot.dt
            i_err = i_err + 3.5e-5 * ang_error / self.robot.dt
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
            self, force: float, T_Base2Socket: Transformation, axis: str = 'z', time_out: float = 10.0) -> bool:
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
            self.robot.motion_mode(target)
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
