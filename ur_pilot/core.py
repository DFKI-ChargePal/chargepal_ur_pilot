from __future__ import annotations

# global
import math
import numpy as np
from pathlib import Path
from enum import auto, Enum
import chargepal_aruco as ca
from time import perf_counter
from contextlib import contextmanager
from rigmopy import Pose, Vector3d, Vector6d, Transformation

# local
from ur_pilot import utils
from ur_pilot.robot import Robot

# typing
from typing import Iterator, Sequence


@contextmanager
def connect(config: Path | None = None) -> Iterator[Pilot]:
    pilot = Pilot(config)
    try:
        yield pilot
    finally:
        pilot.disconnect()


class ControlContext(Enum):
    DISABLED = auto()
    FORCE = auto()
    SERVO = auto()
    MOTION = auto()
    HYBRID = auto()
    POSITION = auto()
    VELOCITY = auto()
    TEACH_IN = auto()


class Pilot:

    def __init__(self, config: Path | None = None) -> None:
        """ Core class to interact with the robot

        Args:
            config: Path to a configuration toml file

        Raises:
            FileNotFoundError: Check if configuration file exists
        """
        if config is not None and not config.exists():
            raise FileNotFoundError(f"Configuration with file path {config} not found.")
        self.robot = Robot(config)
        self.control_context = ControlContext.DISABLED

    def _check_control_context(self, expected: ControlContext | list[ControlContext]) -> None:
        if self.control_context is ControlContext.DISABLED:
            raise RuntimeError(f"Pilot is not in any control context. Running actions is not possible.")
        if type(expected) == list and self.control_context not in expected:
            raise RuntimeError(f"This action is not able to use one of the control context '{self.control_context}'")
        if type(expected) == ControlContext and self.control_context is not expected:
            raise RuntimeError(f"This action is not able to use the control context '{self.control_context}'")

    def exit_control_context(self) -> None:
        if self.control_context == ControlContext.POSITION:
            pass
        elif self.control_context == ControlContext.SERVO:
            self.robot.stop_servoing()
        elif self.control_context == ControlContext.FORCE:
            self.robot.stop_force_mode()
        elif self.control_context == ControlContext.MOTION:
            self.robot.stop_motion_mode()
        elif self.control_context == ControlContext.HYBRID:
            self.robot.stop_hybrid_mode()
        elif self.control_context == ControlContext.TEACH_IN:
            self.robot.stop_teach_mode()
        self.control_context = ControlContext.DISABLED

    def enter_position_control(self) -> None:
        self.control_context = ControlContext.POSITION

    @contextmanager
    def position_control(self) -> Iterator[None]:
        self.enter_position_control()
        yield
        self.exit_control_context()

    def enter_servo_control(self) -> None:
        self.control_context = ControlContext.SERVO

    @contextmanager
    def servo_control(self) -> Iterator[None]:
        self.enter_servo_control()
        yield
        self.exit_control_context()

    def enter_force_control(self, gain: float | None = None, damping: float | None = None) -> None:
        self.robot.set_up_force_mode(gain=gain, damping=damping)
        self.control_context = ControlContext.FORCE

    @contextmanager
    def force_control(self, gain: float | None = None, damping: float | None = None) -> Iterator[None]:
        self.enter_force_control(gain=gain, damping=damping)
        yield
        self.exit_control_context()

    def enter_motion_control(self,
                             error_scale: float | None = None,
                             force_limit: float | None = None,
                             Kp_6: Sequence[float] | None = None,
                             Kd_6: Sequence[float] | None = None,
                             ft_gain: float | None = None,
                             ft_damping: float | None = None) -> None:
        self.robot.set_up_motion_mode(
            error_scale=error_scale,
            force_limit=force_limit,
            Kp_6=Kp_6, Kd_6=Kd_6,
            ft_gain=ft_gain,
            ft_damping=ft_damping)
        self.control_context = ControlContext.MOTION

    @contextmanager
    def motion_control(self,
                       error_scale: float | None = None,
                       force_limit: float | None = None,
                       Kp_6: Sequence[float] | None = None,
                       Kd_6: Sequence[float] | None = None,
                       ft_gain: float | None = None,
                       ft_damping: float | None = None) -> Iterator[None]:
        self.enter_motion_control(
            error_scale=error_scale,
            force_limit=force_limit,
            Kp_6=Kp_6, Kd_6=Kd_6,
            ft_gain=ft_gain,
            ft_damping=ft_damping)
        yield
        self.exit_control_context()

    def enter_hybrid_control(self,
                             error_scale: float | None = None,
                             force_limit: float | None = None,
                             Kp_6_force: Sequence[float] | None = None,
                             Kd_6_force: Sequence[float] | None = None,
                             Kp_6_motion: Sequence[float] | None = None,
                             Kd_6_motion: Sequence[float] | None = None,
                             ft_gain: float | None = None,
                             ft_damping: float | None = None) -> None:
        self.robot.set_up_hybrid_mode(
            error_scale=error_scale,
            force_limit=force_limit,
            Kp_6_force=Kp_6_force,
            Kd_6_force=Kd_6_force,
            Kp_6_motion=Kp_6_motion,
            Kd_6_motion=Kd_6_motion,
            ft_gain=ft_gain,
            ft_damping=ft_damping)
        self.control_context = ControlContext.MOTION

    @contextmanager
    def hybrid_control(self,
                       error_scale: float | None = None,
                       force_limit: float | None = None,
                       Kp_6_force: Sequence[float] | None = None,
                       Kd_6_force: Sequence[float] | None = None,
                       Kp_6_motion: Sequence[float] | None = None,
                       Kd_6_motion: Sequence[float] | None = None,
                       ft_gain: float | None = None,
                       ft_damping: float | None = None) -> Iterator[None]:
        self.enter_hybrid_control(
            error_scale=error_scale,
            force_limit=force_limit,
            Kp_6_force=Kp_6_force,
            Kd_6_force=Kd_6_force,
            Kp_6_motion=Kp_6_motion,
            Kd_6_motion=Kd_6_motion,
            ft_gain=ft_gain,
            ft_damping=ft_damping)
        yield
        self.exit_control_context()

    def enter_teach_in_control(self) -> None:
        self.robot.set_up_teach_mode()
        self.control_context = ControlContext.TEACH_IN

    @contextmanager
    def teach_in_control(self) -> Iterator[None]:
        self.enter_teach_in_control()
        yield
        self.exit_control_context()

    def move_home(self) -> list[float]:
        self._check_control_context(expected=ControlContext.POSITION)
        self.robot.move_home()
        new_j_pos = self.robot.get_joint_pos()
        return new_j_pos

    def move_to_joint_pos(self, q: Sequence[float]) -> list[float]:
        self._check_control_context(expected=ControlContext.POSITION)
        # Move to requested joint position
        self.robot.move_j(q)
        new_joint_pos = self.robot.get_joint_pos()
        return new_joint_pos

    def move_to_tcp_pose(self, target: Pose, time_out: float = 3.0) -> tuple[bool, Pose]:
        self._check_control_context(expected=[ControlContext.POSITION, ControlContext.MOTION])
        fin = False
        # Move to requested TCP pose
        if self.control_context == ControlContext.POSITION:
            self.robot.move_l(target)
            fin = True
        if self.control_context == ControlContext.MOTION:
            t_start = perf_counter()
            tgt_3pts = target.to_3pt_set()
            while True:
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
        self._check_control_context(expected=ControlContext.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        wrench = Vector6d().from_Vector3d(force, Vector3d()).xyzXYZ
        x_ref = np.array(X_tcp.xyz, dtype=np.float32)
        t_start = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=wrench)
            if (perf_counter() - t_start) > duration:
                break
            if ca.EventObserver.state is ca.EventObserver.Type.QUIT:
                break
        x_now = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
        dist: float = np.sum(np.abs(x_now - x_ref))  # L1 norm
        # Stop robot movement.
        self.robot.force_mode(task_frame=task_frame, selection_vector=6 * [0], wrench=6 * [0.0])
        return dist

    def plug_in_force_mode(self, axis: str, force: float, time_out: float) -> bool:
        self._check_control_context(expected=ControlContext.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        x_ref = np.array(X_tcp.xyz, dtype=np.float32)
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
            if ca.EventObserver.state is ca.EventObserver.Type.QUIT:
                break
        # Stop robot movement.
        self.robot.force_mode(task_frame=task_frame, selection_vector=6*[0], wrench=6*[0.0])
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
        self._check_control_context(expected=ControlContext.FORCE)
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
        self._check_control_context(expected=ControlContext.FORCE)
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
                task_frame=task_frame,
                selection_vector=select_vec,
                wrench=wrench_vec
            )
            # Get current transformation from base to end-effector
            T_Base2Tip = Transformation().from_pose(self.robot.get_pose('tool_tip'))
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
        self._check_control_context(expected=ControlContext.FORCE)
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
                task_frame=task_frame,
                selection_vector=select_vec,
                wrench=wrench
            )
            # Get current transformation from base to end-effector
            T_Base2Tip = Transformation().from_pose(self.robot.get_pose('tool_tip'))
            T_Tip2Socket = T_Base2Tip.inverse() @ T_Base2Socket
            if T_Tip2Socket.tau[2] <= -0.015:
                fin = True
                break
            t_now = perf_counter()
            if t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement.
        self.robot.force_mode(task_frame=task_frame, selection_vector=6*[0], wrench=6*[0.0])
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
        self._check_control_context(expected=ControlContext.FORCE)
        # wrench parameter
        freq = 10.0
        f = abs(force)
        m = abs(moment)
        # Wrench will be applied with respect to the current TCP pose
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        select_vec = [0, 0, 1, 0, 1, 0]
        fin = False
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
            # Get current transformation from base to end-effector
            T_Base2Tip = Transformation().from_pose(self.robot.get_pose('tool_tip'))
            T_Tip2Socket = T_Base2Tip.inverse() @ T_Base2Socket
            if T_Tip2Socket.tau[2] <= -0.032:
                fin = True
                break
            t_now = perf_counter()
            if t_now - t_start > time_out:
                fin = False
                break
        # Stop robot movement
        self.robot.force_mode(task_frame=task_frame, selection_vector=6*[0], wrench=6*[0.0])
        return fin

    def plug_out_force_mode(self,
                            wrench: Vector6d, 
                            compliant_axes: list[int], 
                            distance: float, 
                            time_out: float) -> bool:
        self._check_control_context(expected=ControlContext.FORCE)
        # Wrench will be applied with respect to the current TCP pose
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        fin = False
        t_start = perf_counter()
        x_ref = np.array(X_tcp.xyz, dtype=np.float32)
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
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
        self.robot.force_mode(task_frame=task_frame, selection_vector=6*[0], wrench=6*[0.0])
        return fin

    def find_contact_point(self, direction: Sequence[int], time_out: float) -> tuple[bool, Pose]:
        self._check_control_context(expected=ControlContext.FORCE)
        # Map direction to wrench
        wrench = np.clip([10.0 * d for d in direction], -10.0, 10.0).tolist()
        selection_vector = [1 if d != 0 else 0 for d in direction]
        task_frame = 6 * [0.0]  # Robot base
        fin = False
        x_ref = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
        t_start = t_ref = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
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
        return fin, self.robot.get_pose(frame='flange')

    def sensing_depth(self, T_Base2Target: Transformation, time_out: float) -> tuple[bool, Transformation]:
        self._check_control_context(expected=ControlContext.FORCE)
        # Parameter set. Sensing is in tool direction
        selection_vector = [0, 0, 1, 0, 0, 0]
        wrench = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        # Process observation variables
        fin = False
        x_ref = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
        t_start = t_ref = perf_counter()
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
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
        self._check_control_context(expected=ControlContext.MOTION)
        fin = False
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
        self.robot.force_mode(6*[0.0], 6*[0], 6*[0.0])
        return fin, self.robot.get_tcp_pose()

    def relax(self, time_duration: float) -> Pose:
        self._check_control_context(expected=ControlContext.FORCE)

        X_tcp = self.robot.get_tcp_pose()
        task_frame = X_tcp.xyz + X_tcp.axis_angle
        t_start = perf_counter()
        # Apply zero wrench and be compliant in all axes
        wrench = 6 * [0.0]
        compliant_axes = 6 * [1]
        while perf_counter() - t_start < time_duration:
            self.robot.force_mode(
                task_frame=task_frame,
                selection_vector=compliant_axes,
                wrench=wrench)
        # Stop robot movement.
        self.robot.force_mode(task_frame=task_frame, selection_vector=6*[0], wrench=6*[0.0])
        return self.robot.get_tcp_pose()

    def retreat(self, task_frame: Sequence[float], direction: Sequence[int], distance: float = 0.02, time_out: float = 3.0) -> tuple[bool, Pose]:
        self._check_control_context(expected=ControlContext.FORCE)
        # Map direction to wrench
        wrench = np.clip([10.0 * d for d in direction], -10.0, 10.0).tolist()
        selection_vector = [1 if d != 0 else 0 for d in direction]
        fin = False
        t_start = perf_counter()
        x_ref = np.array(self.robot.get_tcp_pose().xyz, dtype=np.float32)
        while True:
            # Apply wrench
            self.robot.force_mode(
                task_frame=task_frame,
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

    def move_joints_random(self) -> list[float]:
        self._check_control_context(expected=ControlContext.POSITION)
        # Move to home joint position
        self.robot.move_home()
        # Move to random joint positions near to the home configuration
        home_q = np.array(self.robot.home_joint_config, dtype=np.float32)
        tgt_joint_q: list[float] = (home_q + (np.random.rand(6) * 2.0 - 1.0) * 0.075).tolist()
        self.robot.move_j(tgt_joint_q)
        # Move back to home joint positions
        self.robot.move_home()
        return tgt_joint_q

    def disconnect(self) -> None:
        """ Exit function which will be called from the context manager at the end """
        self.exit_control_context()
        self.robot.exit()
