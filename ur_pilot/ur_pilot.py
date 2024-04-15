from __future__ import annotations

# libs
import math
import time
import cvpd as pd
import numpy as np
from pathlib import Path
import spatialmath as sm
from strenum import StrEnum
from time import perf_counter
from contextlib import contextmanager

import ur_pilot.utils
from ur_pilot import utils
from ur_pilot import config
from ur_pilot.ur_robot import URRobot
from ur_pilot.config_mdl import Config, read_toml
from ur_pilot.control_mode import ControlContextManager
from ur_pilot.end_effector.bota_sensor import BotaFtSensor
from ur_pilot.end_effector.flange_eye_calibration import FlangeEyeCalibration
from ur_pilot.end_effector.models import CameraModel, PlugModel, TwistCouplingModel, BotaSensONEModel

# typing
from numpy import typing as npt
from camera_kit import CameraBase
from typing import Iterator, Sequence


_t_now = perf_counter


class EndEffectorFrames(StrEnum):

    FLANGE = 'flange'
    FT_SENSOR = 'ft_sensor'
    COUPLING_SAFETY = 'coupling_safety'
    COUPLING_LOCKED = 'coupling_locked'
    COUPLING_UNLOCKED = 'coupling_unlocked'
    PLUG_LIP = 'plug_lip'
    PLUG_TIP = 'plug_tip'
    PLUG_SENSE = 'plug_sense'
    PLUG_SAFETY = 'plug_safety'
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
        self.coupling_model = TwistCouplingModel(**self.cfg.pilot.coupling.dict())
        self.plug_model = PlugModel(self.cfg)
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

    def find_target_pose(self,
                         detector_fp: str | Path,
                         max_samples: int = 10,
                         time_out: float = 4.0,
                         render: bool = False) -> tuple[bool, sm.SE3]:
        """ Helper method to find pose of a target object. Object described by a detector configuration

        Args:
            detector_fp: File path to the detector configuration
            max_samples: Maximum number of samples used to average target pose measurement
            time_out:    Max time for detection
            render:      Flag to show detection results on screen

        Returns:
            (valid result flag | Averaged pose estimation w.r.t robot base)
        """
        search_rate = 0.5 * time_out/max_samples
        n_max, t_max = int(abs(max_samples)), abs(time_out)
        T_base2target_meas: sm.SE3 | None = None
        detector = pd.factory.create(Path(detector_fp))
        if self.cam is None:
            raise RuntimeError(f"No registered camera. Detection cannot be executed without it.")
        detector.register_camera(self.cam)
        t_start = _t_now()
        for _ in range(n_max):
            time.sleep(search_rate)
            found, T_cam2target = detector.find_pose(render=render)
            if found:
                # Get transformation matrices
                T_flange2cam = self.cam_mdl.T_flange2camera
                T_base2flange = self.get_pose(EndEffectorFrames.FLANGE)
                # Get searched transformation
                if T_base2target_meas is None:
                    T_base2target_meas = T_base2flange * T_flange2cam * T_cam2target
                else:
                    T_base2target_meas.append(T_base2flange * T_flange2cam * T_cam2target)
            # Check for time boundary
            if _t_now() - t_start > t_max:
                break
        if T_base2target_meas is None:
            valid_result, T_base2target = False, sm.SE3()
        elif len(T_base2target_meas) == 1:
            valid_result, T_base2target = True, T_base2target_meas
        else:
            q_avg = ur_pilot.utils.quatAvg(sm.UnitQuaternion(T_base2target_meas))
            t_avg = np.mean(T_base2target_meas.t, axis=0)
            T_base2target = sm.SE3().Rt(R=q_avg.SO3(), t=t_avg)
            valid_result = True
        return valid_result, T_base2target

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
        elif frame == EndEffectorFrames.COUPLING_SAFETY:
            if self.extern_sensor:
                offset = self.ft_sensor_mdl.T_mounting2wrench * self.coupling_model.T_mounting2safety
            else:
                offset = self.coupling_model.T_mounting2safety
        elif frame == EndEffectorFrames.COUPLING_LOCKED:
            if self.extern_sensor:
                offset = self.ft_sensor_mdl.T_mounting2wrench * self.coupling_model.T_mounting2locked
            else:
                offset = self.coupling_model.T_mounting2locked
        elif frame == EndEffectorFrames.COUPLING_UNLOCKED:
            if self.extern_sensor:
                offset = self.ft_sensor_mdl.T_mounting2wrench * self.coupling_model.T_mounting2unlocked
            else:
                offset = self.coupling_model.T_mounting2unlocked
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
        elif frame == EndEffectorFrames.PLUG_SAFETY:
            if self.extern_sensor:
                offset = self.ft_sensor_mdl.T_mounting2wrench * self.plug_model.T_mounting2safety
            else:
                offset = self.plug_model.T_mounting2safety
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
        f_tool_wrt_world = self.coupling_model.f_inertia
        f_tool_wrt_ft: npt.NDArray[np.float_] = self.rot_world2arm * f_tool_wrt_world

        def cross(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float32]:
            # Overwrite numpy cross function to avoid errors by IDE
            a_ = np.array(a, dtype=np.float32)
            b_ = np.array(b, dtype=np.float32)
            return np.cross(a_, b_)

        t_tool_wrt_ft = cross(self.coupling_model.com, f_tool_wrt_ft)
        ft_comp = sm.SpatialForce(np.append(f_tool_wrt_ft, t_tool_wrt_ft))
        return ft_raw + ft_comp

    def set_tcp(self, frame: EndEffectorFrames | str = 'flange') -> None:
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
            t_start = _t_now()
            tgt_3pts = utils.se3_to_3pt_set(target)
            while True:
                # self.robot.motion_controller.update(target_se3)
                self.robot.motion_mode(target)
                cur_3pts = utils.se3_to_3pt_set(self.robot.tcp_pose)
                error = np.mean(np.abs(tgt_3pts, cur_3pts))
                if error <= 0.005:  # 5 mm
                    fin = True
                    break
                elif _t_now() - t_start > time_out:
                    fin = False
                    break
        return fin, self.robot.tcp_pose

    def push_linear(self, force: npt.ArrayLike, compliant_axes: list[int], duration: float) -> float:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        x_ref = self.robot.tcp_pos
        task_frame = self.robot.tcp_pose
        wrench = np.append(np.reshape(force, 3), np.zeros(3)).tolist()
        t_start = _t_now()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=wrench)
            if (_t_now() - t_start) > duration:
                break
        x_now = self.robot.tcp_pos
        dist: float = np.sum(np.abs(x_now - x_ref))  # L1 norm
        # Stop robot movement.
        self.robot.pause_force_mode()
        return dist

    def screw_ee_force_mode(self, torque: float, ang: float, time_out: float) -> bool:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Limit input
        torque = np.clip(torque, 0.0, 5.0)
        wrench_vec = 6 * [0.0]
        compliant_axes = [0, 0, 1, 0, 0, 1]
        # Wrench will be applied with respect to the current TCP pose
        task_frame = self.get_pose(EndEffectorFrames.FLANGE)
        # Create target
        ee_jt_pos = self.robot.joint_pos[-1]
        ee_jt_pos_tgt = ee_jt_pos + ang
        # Setup controller
        screw_ctrl = utils.PIDController(kp=1.0, kd=1.0, ki=1.0)
        # Exit criteria parameter
        success, t_start = False, _t_now()
        while not success:
            # Angular error
            ee_jt_pos_now = self.robot.joint_pos[-1]
            ang_error = float(ee_jt_pos_tgt - ee_jt_pos_now)
            wrench_vec[-1] = np.clip(screw_ctrl.update(ang_error, self.robot.dt), -torque, torque)
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=wrench_vec)
            if abs(ang_error) < 5e-3:
                success = True
            if _t_now() - t_start > time_out:
                break
        # Stop robot movement.
        self.relax(0.5)
        self.robot.pause_force_mode()
        return success

    def try2_approach_to_plug(self, T_base2socket: sm.SE3) -> tuple[bool, tuple[float, float]]:
        self.context.check_mode(expected=self.context.mode_types.POSITION)
        # Set TCP offset
        self.set_tcp(EndEffectorFrames.COUPLING_SAFETY)
        # Get estimation of the plug pose
        T_socket2mounting = self.plug_model.T_mounting2lip.inv()
        T_mounting2plug = self.coupling_model.T_mounting2locked
        T_base2plug = T_base2socket * T_socket2mounting * T_mounting2plug
        success, _ = self.move_to_tcp_pose(T_base2plug)
        # Evaluate spatial error
        T_base2plug_meas = self.get_pose(EndEffectorFrames.COUPLING_SAFETY)
        lin_ang_err = utils.lin_rot_error(T_base2plug, T_base2plug_meas)
        return success, lin_ang_err

    def try2_approach_to_socket(self, T_base2socket: sm.SE3) -> tuple[bool, tuple[float, float]]:
        self.context.check_mode(expected=self.context.mode_types.POSITION)
        # Set TCP offset
        self.set_tcp(EndEffectorFrames.PLUG_SAFETY)
        success, _ = self.move_to_tcp_pose(T_base2socket)
        # Evaluate spatial error
        T_base2socket_meas = self.get_pose(EndEffectorFrames.PLUG_SAFETY)
        lin_ang_err = utils.lin_rot_error(T_base2socket, T_base2socket_meas)
        return success, lin_ang_err

    def try2_couple_to_plug(self, T_base2socket: sm.SE3, time_out: float = 10.0) -> tuple[bool, tuple[float, float]]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Limit input
        time_out = abs(time_out)
        # Setup controller
        wrench_vec = 6 * [0.0]
        selection_vec = [1, 1, 1, 0, 0, 1]
        x_ctrl = utils.PDController(kp=100.0, kd=0.99)
        y_ctrl = utils.PDController(kp=100.0, kd=0.99)
        z_ctrl = utils.PIDController(kp=500.0, kd=0.99, ki=1000.0)
        yaw_ctrl = utils.PDController(kp=10.0, kd=0.99)

        # Get estimation of the plug pose
        T_socket2mounting = self.plug_model.T_mounting2lip.inv()
        T_mounting2plug = self.coupling_model.T_mounting2locked
        T_base2plug_est = T_base2socket * T_socket2mounting * T_mounting2plug
        # T_socket2plug = self.plug_model.T_mounting2lip.inv()
        # T_base2plug_est = T_base2socket * T_socket2plug

        # Wrench will be applied with respect to the current TCP pose
        task_frame = T_base2plug_meas = self.get_pose(EndEffectorFrames.COUPLING_UNLOCKED)
        T_meas2est = T_base2plug_meas.inv() * T_base2plug_est
        p_meas2est_ref = p_meas2est = T_meas2est.t
        yaw_meas2est_ref = yaw_meas2est = float(sm.UnitQuaternion(sm.SO3(T_meas2est.R).norm()).rpy(order='zyx')[-1])
        success = False
        t_ref = t_now = t_start = _t_now()
        while t_now - t_start <= time_out:
            # Update controller
            wrench_vec[0] = np.clip(x_ctrl.update(p_meas2est[0], self.robot.dt), -50.0, 50.0)
            wrench_vec[1] = np.clip(y_ctrl.update(p_meas2est[1], self.robot.dt), -50.0, 50.0)
            wrench_vec[2] = np.clip(z_ctrl.update(p_meas2est[2], self.robot.dt), -50.0, 50.0)
            wrench_vec[-1] = np.clip(yaw_ctrl.update(yaw_meas2est, self.robot.dt), -4.0, 4.0)
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=selection_vec,
                wrench=wrench_vec,
            )
            # Update error
            T_base2plug_meas = self.get_pose(EndEffectorFrames.COUPLING_UNLOCKED)
            T_meas2est = T_base2plug_meas.inv() * T_base2plug_est
            p_meas2est = T_meas2est.t
            yaw_meas2est = float(sm.UnitQuaternion(sm.SO3(T_meas2est.R).norm()).rpy(order='zyx')[-1])
            t_now = _t_now()
            # Check every second if robot is still moving
            if t_now - t_ref > 1.0:
                if (np.allclose(p_meas2est_ref, p_meas2est, atol=0.002)
                        and np.isclose(yaw_meas2est_ref, yaw_meas2est, atol=0.015)):
                    # Check whether couple depth is reached
                    d_err = p_meas2est[2]
                    if abs(d_err) <= 0.005:
                        success = True
                    else:
                        success = False
                    break
                t_ref, p_meas2est_ref = t_now, p_meas2est
            # Check whether error is getting to large
            T_base2plug_meas = self.get_pose(EndEffectorFrames.COUPLING_UNLOCKED)
            T_meas2est = T_base2plug_meas.inv() * T_base2plug_est
            xy_error = float(np.linalg.norm(T_meas2est.t[0:2]))
            ang_error = float(T_base2plug_est.angdist(T_base2plug_meas))
            if xy_error > 0.02 or ang_error > np.pi/12:  # xy limit 2 cm angular limit 15 deg
                success = False
                break
        # Stop robot movement.
        self.robot.pause_force_mode()
        # Evaluate spatial error
        lin_ang_err = utils.lin_rot_error(T_base2plug_est, T_base2plug_meas)
        return success, lin_ang_err

    def try2_decouple_to_plug(self, time_out: float = 5.0) -> tuple[bool, tuple[float, float]]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Limit input arguments
        time_out = abs(time_out)
        # Setup controller
        wrench_vec = 6 * [0.0]
        selection_vec = [0, 0, 1, 0, 0, 1]
        yaw_ctrl = utils.PDController(kp=10.0, kd=0.99)
        z_ctrl = utils.PIDController(kp=500.0, kd=0.99, ki=1000.0)

        # Get estimation of the safety pose
        T_base2safety_est = self.get_pose(EndEffectorFrames.COUPLING_UNLOCKED)
        # Get current coupling pose
        task_frame = T_base2safety_meas = self.get_pose(EndEffectorFrames.COUPLING_SAFETY)
        # Get control input difference
        T_meas2est = T_base2safety_meas.inv() * T_base2safety_est
        p_meas2est = T_meas2est.t
        yaw_meas2est = float(sm.UnitQuaternion(sm.SO3(T_meas2est.R).norm()).rpy(order='zyx')[-1])
        success = False
        t_now = t_start = _t_now()
        while t_now - t_start <= time_out:
            # Update controller
            wrench_vec[2] = np.clip(z_ctrl.update(p_meas2est[2], self.robot.dt), -50.0, 50.0)
            wrench_vec[-1] = np.clip(yaw_ctrl.update(yaw_meas2est, self.robot.dt), -4.0, 4.0)
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=selection_vec,
                wrench=wrench_vec,
            )
            # Update error
            T_base2safety_meas = self.get_pose(EndEffectorFrames.COUPLING_SAFETY)
            T_meas2est = T_base2safety_meas.inv() * T_base2safety_est
            yaw_meas2est = float(sm.UnitQuaternion(sm.SO3(T_meas2est.R).norm()).rpy(order='zyx')[-1])
            p_meas2est = T_meas2est.t
            if abs(p_meas2est[2]) <= 0.003:  # Only check for depth value of the coupling
                success = True
                break
            t_now = _t_now()
        # Stop robot movement
        self.robot.pause_force_mode()
        # Evaluate spatial error
        lin_ang_err = utils.lin_rot_error(T_base2safety_est, T_base2safety_meas)
        return success, lin_ang_err

    def try2_unlock_plug(self, T_base2socket: sm.SE3, time_out: float = 12.0) -> tuple[bool, tuple[float, float]]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Get estimation of the plug pose
        T_socket2plug = self.plug_model.T_mounting2lip.inv()
        T_base2plug_est = T_base2socket * T_socket2plug
        # Turn end effector by 90 degrees counter-clockwise
        success = self.screw_ee_force_mode(torque=4.0, ang=-np.pi/2, time_out=time_out)
        # Evaluate spatial error
        T_base2plug_meas = self.get_pose(EndEffectorFrames.COUPLING_UNLOCKED)
        lin_ang_err = utils.lin_rot_error(T_base2plug_est, T_base2plug_meas)
        return success, lin_ang_err

    def try2_lock_plug(self, T_base2socket: sm.SE3, time_out: float = 12.0) -> tuple[bool, tuple[float, float]]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Get estimation of the plug pose
        T_socket2plug = self.plug_model.T_mounting2lip.inv()
        T_base2plug_est = T_base2socket * T_socket2plug
        # Turn end effector by 90 degrees clockwise
        success = self.screw_ee_force_mode(torque=4.0, ang=np.pi/2, time_out=time_out)
        # Evaluate spatial error
        T_base2plug_meas = self.get_pose(EndEffectorFrames.COUPLING_LOCKED)
        lin_ang_err = utils.lin_rot_error(T_base2plug_est, T_base2plug_meas)
        return success, lin_ang_err

    def try2_engage_with_socket(self, T_base2socket: sm.SE3, time_out: float = 6.0) -> tuple[bool, tuple[float, float]]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Limit input
        time_out = abs(time_out)
        # Setup controller
        wrench_vec = 6 * [0.0]
        selection_vec = [1, 1, 1, 0, 0, 1]
        x_ctrl = utils.PDController(kp=100.0, kd=0.99)
        y_ctrl = utils.PDController(kp=100.0, kd=0.99)
        z_ctrl = utils.PIDController(kp=500.0, kd=0.99, ki=1000.0)
        yaw_ctrl = utils.PDController(kp=10.0, kd=0.99)
        # Get estimation of the engaged socket pose
        T_base2engaged_est = T_base2socket * sm.SE3.Tz(0.015)
        # Wrench will be applied with respect to the current TCP pose
        task_frame = T_base2tip_meas = self.get_pose(EndEffectorFrames.PLUG_TIP)
        T_tip2engaged = T_base2tip_meas.inv() * T_base2engaged_est
        xyz_tip2engaged = T_tip2engaged.t
        yaw_tip2engaged = float(sm.UnitQuaternion(sm.SO3(T_tip2engaged.R).norm()).rpy(order='zyx')[-1])
        success = False
        t_ref = t_now = t_start = _t_now()
        while t_now - t_start <= time_out:
            # Update controller
            wrench_vec[0] = np.clip(x_ctrl.update(xyz_tip2engaged[0], self.robot.dt), -50.0, 50.0)
            wrench_vec[1] = np.clip(y_ctrl.update(xyz_tip2engaged[1], self.robot.dt), -50.0, 50.0)
            wrench_vec[2] = np.clip(z_ctrl.update(xyz_tip2engaged[2], self.robot.dt), -50.0, 50.0)
            wrench_vec[-1] = np.clip(yaw_ctrl.update(yaw_tip2engaged, self.robot.dt), -4.0, 4.0)
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=selection_vec,
                wrench=wrench_vec,
            )
            # Update error
            T_base2tip_meas = self.get_pose(EndEffectorFrames.PLUG_TIP)
            T_tip2engaged = T_base2tip_meas.inv() * T_base2engaged_est
            xyz_tip2engaged = T_tip2engaged.t
            yaw_tip2engaged = float(sm.UnitQuaternion(sm.SO3(T_tip2engaged.R).norm()).rpy(order='zyx')[-1])
            t_now = _t_now()
            # Check if alignment error is not getting to large
            xy_error = utils.lin_error(T_base2engaged_est, T_base2tip_meas, axes='xy')
            screw_error = utils.rot_error_single_axis(T_base2engaged_est, T_base2tip_meas, 'yaw')
            if xy_error > 0.025 or screw_error > 0.1:
                break
            # Check on whether the engaging error is small enough
            z_error = utils.lin_error(T_base2engaged_est, T_base2tip_meas, axes='z')
            if z_error < 2.5e-3:
                success = True
                break
        # Evaluate spatial error
        T_base2tip_meas = self.get_pose(EndEffectorFrames.PLUG_TIP)
        lin_ang_err = utils.lin_rot_error(T_base2engaged_est, T_base2tip_meas)
        return success, lin_ang_err

    def try2_insert_plug(self, T_base2socket: sm.SE3, time_out: float = 10.0) -> tuple[bool, tuple[float, float]]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Inputs
        f_start, f_end = 85.0, 125.0
        wrench_vec = 6 * [0.0]
        selection_vec = [1, 1, 1, 0, 0, 1]
        T_base2socket_est = T_base2socket

        force_ramp = utils.ramp(f_start, f_end, time_out)
        success = False
        for f in force_ramp:
            wrench_vec[2] = f
            task_frame = T_base2socket_meas_ref = self.get_pose(EndEffectorFrames.PLUG_LIP)
            # lin_ang_err_ref = utils.lin_rot_error(T_base2socket_est, T_base2socket_meas_ref)
            _t_start = _t_now()
            while _t_now() - _t_start <= 1.0:
                # Apply wrench
                self.robot.force_mode(
                    task_frame=task_frame,
                    selection_vector=selection_vec,
                    wrench=wrench_vec)
            self.relax(0.5)
            self.robot.pause_force_mode()
            T_base2socket_meas = self.get_pose(EndEffectorFrames.PLUG_LIP)
            xy_error = utils.lin_error(T_base2socket_est, T_base2socket_meas, 'xy')
            screw_error = utils.rot_error_single_axis(T_base2socket_est, T_base2socket_meas, 'yaw')
            if xy_error > 0.025 or screw_error > 0.1:
                break
            z_error = utils.lin_error(T_base2socket_est, T_base2socket_meas, 'z')
            insertion_progress = utils.lin_error(T_base2socket_meas_ref, T_base2socket_meas, 'z')
            if insertion_progress < 0.0025 and z_error < 0.005:
                success = True
                break
        # Evaluate spatial error
        T_base2socket_meas = self.get_pose(EndEffectorFrames.COUPLING_LOCKED)
        lin_ang_err = utils.lin_rot_error(T_base2socket_est, T_base2socket_meas)
        return success, lin_ang_err

    def try2_remove_plug(self, time_out: float = 10.0) -> tuple[bool, tuple[float, float]]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Limit input arguments
        time_out = abs(time_out)
        # Setup controller
        wrench_vec = 6 * [0.0]
        selection_vec = [0, 0, 1, 0, 0, 0]
        depth_ctrl = utils.PIDController(kp=500.0, kd=0.99, ki=1000.0)
        # Get estimation of the safety pose
        T_base2safety_est = self.get_pose(EndEffectorFrames.PLUG_LIP)
        # Get current plug-in pose
        task_frame = T_base2safety_meas = self.get_pose(EndEffectorFrames.PLUG_SAFETY)
        # Get control input difference
        T_meas2est = T_base2safety_meas.inv() * T_base2safety_est
        p_meas2est = T_meas2est.t
        success = False
        t_now = t_start = _t_now()
        while t_now - t_start <= time_out:
            # Update controller
            wrench_vec[2] = np.clip(depth_ctrl.update(p_meas2est[2], self.robot.dt) - 50.0, -125.0, 0.0)
            self.robot.force_mode(task_frame=task_frame, selection_vector=selection_vec, wrench=wrench_vec)
            # Update error
            T_base2safety_meas = self.get_pose(EndEffectorFrames.PLUG_SAFETY)
            T_meas2est = T_base2safety_meas.inv() * T_base2safety_est
            p_meas2est = T_meas2est.t
            if abs(p_meas2est[2]) <= 0.003:  # Only check for depth value of the coupling
                success = True
                break
            t_now = _t_now()
        # Stop robot movement
        self.robot.pause_force_mode()
        # Evaluate spatial error
        lin_ang_err = utils.lin_rot_error(T_base2safety_est, T_base2safety_meas)
        return success, lin_ang_err

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
        t_ref = t_start = _t_now()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=wrench_vec)
            t_now = _t_now()
            # Check every second if robot is still moving
            if t_now - t_ref > 1.0:
                x_now = self.robot.tcp_pos
                if np.allclose(x_ref, x_now, atol=0.001):
                    fin = True
                    break
                t_ref, x_ref = t_now, x_now
            if t_now - t_start > time_out:
                break
        # Stop robot movement
        self.robot.pause_force_mode()
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
        t_start = _t_now()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=select_vec,
                wrench=wrench_vec
            )
            # Get current transformation from base to end-effector
            T_Base2Tip = self.get_pose('tool_tip')
            T_Tip2Socket = T_Base2Tip.inv() * T_Base2Socket
            if T_Tip2Socket.t[wrench_idx] <= -0.032:
                fin = True
                break
            t_now = _t_now()
            if t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.pause_force_mode()
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
        t_start = _t_now()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=select_vec,
                wrench=wrench
            )
            # Get current transformation from base to end-effector
            T_Base2Tip = self.get_pose('tool_tip')
            T_Tip2Socket = T_Base2Tip.inv() * T_Base2Socket
            if T_Tip2Socket.tau[2] <= -0.015:
                fin = True
                break
            t_now = _t_now()
            if t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.pause_force_mode()
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
        t_start = _t_now()
        while True:
            dt = _t_now() - t_start
            # wrench = [0.0, 0.0, f, m * math.sin(freq * dt), m * math.cos(freq * dt), 0.0]
            wrench = [0.0, 0.0, f, 0.0, m * math.sin(freq * dt), 0.0]
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=select_vec,
                wrench=wrench
            )
            # Get current transformation from base to end-effector
            T_Base2Tip = self.get_pose('tool_tip')
            T_Tip2Socket = T_Base2Tip.inv() * T_Base2Socket
            if T_Tip2Socket.tau[2] <= -0.032:
                fin = True
                break
            t_now = _t_now()
            if t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.pause_force_mode()
        return fin

    def tcp_force_mode(self,
                       wrench: npt.ArrayLike, compliant_axes: list[int], distance: float, time_out: float) -> bool:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        task_frame = self.robot.tcp_pose
        t_start = _t_now()
        x_ref = self.robot.tcp_pos
        ft_vec = np.reshape(wrench, 6).tolist()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=ft_vec)
            x_now = self.robot.tcp_pos
            l2_norm_dist = np.linalg.norm(x_now - x_ref)
            t_now = _t_now()
            if l2_norm_dist >= distance:
                fin = True
                break
            elif t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.pause_force_mode()
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
        t_start = _t_now()
        x_ref = self.robot.tcp_pos
        ft_vec = np.reshape(wrench, 6).tolist()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=ft_vec)
            x_now = self.robot.tcp_pos
            l2_norm_dist = np.linalg.norm(x_now - x_ref)
            t_now = _t_now()
            if l2_norm_dist >= distance:
                fin = True
                break
            elif t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.pause_force_mode()
        return fin

    def find_contact_point(self, direction: Sequence[int], time_out: float) -> tuple[bool, sm.SE3]:
        self.context.check_mode(expected=self.context.mode_types.FORCE)
        # Map direction to wrench
        wrench = np.clip([10.0 * d for d in direction], -10.0, 10.0).tolist()
        selection_vector = [1 if d != 0 else 0 for d in direction]
        task_frame = sm.SE3()  # Robot base
        x_ref = self.robot.tcp_pos
        t_start = t_ref = _t_now()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=selection_vector,
                wrench=wrench
            )
            t_now = _t_now()
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
        t_start = t_ref = _t_now()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=selection_vector,
                wrench=wrench
            )
            t_now = _t_now()
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
            # Stop robot movement
            self.robot.pause_force_mode()
            return fin, sm.SE3()
        else:
            # Stop robot movement
            self.robot.pause_force_mode()
            T_Base2Target_meas = sm.SE3.Rt(R=T_Base2Target.R, t=self.robot.tcp_pos)
            return fin, T_Base2Target_meas

    def plug_in_motion_mode(self, target: sm.SE3, time_out: float) -> tuple[bool, sm.SE3]:
        self.context.check_mode(expected=self.context.mode_types.MOTION)
        t_start = _t_now()
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
            elif _t_now() - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.pause_force_mode()
        return fin, self.robot.tcp_pose

    def relax(self, time_duration: float) -> sm.SE3:
        self.context.check_mode(expected=[
            self.context.mode_types.FORCE, self.context.mode_types.MOTION, self.context.mode_types.HYBRID
        ])
        task_frame = self.robot.tcp_pose
        t_start = _t_now()
        # Apply zero wrench and be compliant in all axes
        wrench = 6 * [0.0]
        compliant_axes = 6 * [1]
        while _t_now() - t_start < time_duration:
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=wrench)
        # Stop robot movement.
        self.robot.pause_force_mode()
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
        t_start = _t_now()
        x_ref = self.robot.tcp_pos
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=selection_vector,
                wrench=wrench
            )
            t_now = _t_now()
            x_now = self.robot.tcp_pos
            l2_norm_dist = np.linalg.norm(x_now - x_ref)
            if l2_norm_dist >= distance:
                fin = True
                break
            elif t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.pause_force_mode()
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
