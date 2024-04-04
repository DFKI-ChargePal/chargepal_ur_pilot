""" This file contains common functionality. """
from __future__ import annotations

# libs
import abc
import sys
import pysoem
import numpy as np
import spatialmath as sm

from pathlib import Path

# typing
from typing import Sequence
from numpy import typing as npt


def get_pkg_path() -> Path:
    current_path = Path(__file__).absolute()
    path_parts = list(current_path.parts)
    path_parts.reverse()
    idx = 0
    for i, pp in enumerate(path_parts):
        if pp == 'chargepal_ur_pilot':
            idx = -i
    pkg_path = Path(*current_path.parts[:idx])
    return pkg_path


def check_file_extension(file_path: Path, file_ext: str) -> None:
    if file_path.suffix.strip('.') != file_ext.strip('.'):
        raise ValueError(f"{file_ext.upper()} file with extension .{file_ext} is mandatory. "
                         f"Given file name: {file_path.name}")


def check_file_path(file_path: Path) -> bool:
    """ File exist check with user interaction.

    Args:
        file_path: File path

    Returns:
        True if file does not exist or user has given permission to overwrite.
    """
    if file_path.exists():
        proceed = query_yes_no(f"The file {file_path.name} already exists. Do you want to overwrite it?")
    else:
        proceed = True
    return proceed


def query_yes_no(question: str, default: str = "yes") -> bool:
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def lin_error(T_a2b_ref: sm.SE3, T_a2b_meas: sm.SE3, axes: str = 'xyz') -> float:
    """ Calculate the linear error/distance between two frames

    Args:
        T_a2b_ref:  Reference frame (ground truth)
        T_a2b_meas: Measured frame (prediction)
        axes:       Axes to be considered

    Returns:
        Euclidian distance for the chosen axes
    """
    _valid_axes = ['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz']
    axes = ''.join(sorted(axes.lower()))
    if axes not in _valid_axes:
        raise ValueError(f"Unknown selection for axes: '{axes}'. Valid options are: {_valid_axes}")
    p_meas2ref = (T_a2b_meas.inv() * T_a2b_ref).t
    if 'x' not in axes:
        p_meas2ref[0] = 0.0
    if 'y' not in axes:
        p_meas2ref[1] = 0.0
    if 'z' not in axes:
        p_meas2ref[2] = 0.0
    l_err = float(np.linalg.norm(p_meas2ref))
    return l_err


def rot_error_single_axis(T_a2b_ref: sm.SE3, T_a2b_meas: sm.SE3, axis: str = 'yaw') -> float:
    """ Calculate the rotational error between two frames for one principal axis

    Args:
        T_a2b_ref:  Reference frame (ground truth)
        T_a2b_meas: Measured frame (prediction)
        axis:       Principal axis [roll, pitch, yaw]

    Returns:
        Rotation error for a single principal axis
    """
    _valid_axes_short = ['R', 'P', 'Y']
    _valid_axes_long = ['roll', 'pitch', 'yaw']
    if axis.lower() not in _valid_axes_long and axis.upper() not in _valid_axes_short:
        raise ValueError(f"Unknown principal axis '{axis}'. Valid options are: {_valid_axes_short + _valid_axes_long}")
    axis = axis[0].upper()
    rpy_meas2ref = sm.UnitQuaternion((T_a2b_meas.inv() * T_a2b_ref).R).rpy(order='zyx')
    if axis == 'R':
        r_err = rpy_meas2ref[0]
    elif axis == 'P':
        r_err = rpy_meas2ref[1]
    elif axis == 'Y':
        r_err = rpy_meas2ref[2]
    else:
        raise IndexError(f"Unknown axis index '{axis}'")
    return float(r_err)


def lin_rot_error(T_a2b_est: sm.SE3, T_a2b_meas: sm.SE3) -> tuple[float, float]:
    """ Calculates the linear and rotational error/distance between two frames

    Args:
        T_a2b_est:  Frame of the estimated state
        T_a2b_meas: Frame of the measured state

    Returns:
        Tuple of euclidian distance of the position error and the angular distance of the rotation error
    """
    T_meas2est = T_a2b_meas.inv() * T_a2b_est
    lin_err = float(np.linalg.norm(T_meas2est.t))
    ang_err = float(T_a2b_est.angdist(T_a2b_meas))
    return lin_err, ang_err


def se3_to_str(mat: sm.SE3, digits: int = 4) -> str:
    """ Convert SE(3) object to UR style string
    Args:
        mat:    SE(3) transformation matrix
        digits: Print precision
    
    Returns:
        representation string
    """
    xyz = ", ".join("{0:.{1}f}".format(v, digits) for v in mat.t.tolist())
    aa = ", ".join("{0:.{1}f}".format(v, digits) for v in mat.eulervec().tolist())
    return f"trans=[{xyz}] -- axis ang=[{aa}]"


def se3_to_3pt_set(mat: sm.SE3, dist: float = 1.0, axes: str = 'xy') -> npt.NDArray[np.float64]:
    """ Convert the pose to a set of 3 points, the idea being that, 3 (non-collinear) points can encode both position
        and orientation

    Args:
        mat:  Transformation matrix as SE(3) object
        dist: Distance from original pose to other two points [m].
        axes: Axes of pose orientation to transform other 2 points along.
              Must be a two letter combination of _x, _y and _z, in any order.

    Returns:
        ndarray: A 3x3 np array of each point, with each row containing a single point. The first point is the
                 original point from the 'pose' argument.
    """
    possible_axes = ["xy", "yx", "xz", "zx", "yz", "zy"]
    assert axes in possible_axes, f"param axes must be in: {possible_axes}"
    # First point is just the origin of the pose
    axes = axes.replace('x', '0')
    axes = axes.replace('y', '1')
    axes = axes.replace('z', '2')
    pt1 = mat.t
    tau2, tau3 = np.zeros(3), np.zeros(3)
    tau2[int(axes[0])] = dist
    tau3[int(axes[1])] = dist
    pt2 = np.squeeze(mat * tau2)
    pt3 = np.squeeze(mat * tau3)
    points = np.stack([pt1, pt2, pt3])
    return np.array(points)


def vec_to_str(vec: Sequence[float] | npt.NDArray[np.float32 | np.float64 | np.float_], digits: int = 4) -> str:
    """ Convert a vector to a formatted string
    Args:
        vec:    The vector object
        digits: Print precision

    Returns:
        representation string
    """
    vec_str = ", ".join("{0:.{1}f}".format(v, digits) for v in np.array(vec).flatten().tolist())
    return f"[{vec_str}]"


def quatAvg(Q: sm.UnitQuaternion) -> sm.UnitQuaternion:
    """ Averaging Quaternions.
        Heavily inspired by Dr. Tolga Birdal's work: https://github.com/tolgabirdal/averaging_quaternions
        Based on Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
        "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30, no. 4 (2007): 1193-1197.

    Args:
        Q: UnitQuaternion with N Quaternions.

    Returns:
        The averaged quaternion
    """
    # Form the symmetric accumulator matrix
    A = np.zeros((4, 4))
    N = len(Q)
    w = 1/N

    for i in range(N):
        q = Q[i].vec
        if q[0] < 0:  # handle the antipodal configuration
            q = -q
        A += w * (np.outer(q, q))  # rank 1 update
    # scale
    A /= N
    # Get the eigenvector corresponding to largest eigen value
    return sm.UnitQuaternion(np.linalg.eigh(A)[1][:, -1])


def ramp(start: float, end: float, duration: float) -> list[float]:
    """ Get a list with values from start to end with a step size of one second

    Args:
        start:    Start value of the list
        end:      End value of the list
        duration: Duration in seconds

    Returns:
        A list with values from start to end in a step size of one second
    """
    if start >= end:
        raise ValueError("Start value is greater or equal to end value. This is not allowed")

    n_steps = int(abs(duration))
    if n_steps <= 0:
        raise ValueError("Time period has to be at least one second.")

    step_size = (end - start) / n_steps
    vals = [start + n * step_size for n in range(n_steps)]
    vals.append(float(end))
    return vals


def axis2index(axis: str) -> int:
    """ Map from axis to 6D vector index

    Args:
        axis: Character defining the axis. Possible 'x', 'y', 'z', 'R', 'P' or 'Y'

    Returns:
        Vector index
    """
    if len(axis) != 1:
        raise ValueError('Only one character is allowed as axis parameter')
    a2i = {'x': 0, 'y': 1, 'z': 2, 'R': 3, 'P': 4, 'Y': 5}
    idx = a2i.get(axis)
    if idx is None:
        raise ValueError(f"Unknown axis {axis}! Possible are {a2i.keys()}")
    return idx


def find_ethernet_adapters() -> None:
    adapters = pysoem.find_adapters()

    for i, adapter in enumerate(adapters):
        print(f'Adapter {i}')
        print(f'   {adapter.name}')
        print(f'   {adapter.desc}\n')


class Controller(metaclass=abc.ABCMeta):
    """ Controller superclass. """

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError('Must be implemented in subclass.')

    @abc.abstractmethod
    def update(self, error: float, period: float) -> float:
        raise NotImplementedError('Must be implemented in subclass.')


class PDController(Controller):

    def __init__(self, kp: float = 0.0, kd: float = 0.0) -> None:
        """ One dimension proportional-derivative controller

        Args:
            kp: Proportional gain. Defaults to 0.0.
            kd: Derivative gain. Defaults to 0.0.
        """
        super().__init__()
        self.kp = abs(kp)
        self.kd = abs(kd)
        self.prev_error: float | None = None  # Previous controller error

    def reset(self) -> None:
        """ Reset controller """
        self.prev_error = None

    def update(self, error: float, period: float) -> float:
        """ Controller update step.

        Args:
            error: Controller error
            period: Controller update duration [sec]

        Returns:
            Controller output/system input
        """
        if period > 0.0:
            if self.prev_error is None:
                self.prev_error = error
            out = self.kp * error + self.kd * (error - self.prev_error) / period
            self.prev_error = error
        else:
            out = 0.0
        return out


class PIDController(Controller):

    def __init__(self, kp: float = 0.0, ki: float = 0.0, kd: float = 0.0) -> None:
        """ One dimension proportional-integral-derivative controller

        Args:
            kp: Proportional gain. Defaults to 0.0.
            ki: Integral gain. Defaults to 0.0.
            kd: Derivative gain. Defaults to 0.0.
        """
        super().__init__()
        self.kp = abs(kp)
        self.ki = abs(ki)
        self.kd = abs(kd)
        self.integral = 0.0
        self.prev_error: float | None = None  # Previous controller error

    def reset(self) -> None:
        """ Reset controller """
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, period: float) -> float:
        """ Controller update step.

        Args:
            error: Controller error
            period: Controller update duration [sec]

        Returns:
            Controller output / system input
        """
        if period > 0.0:
            if self.prev_error is None:
                self.prev_error = error
            # Calculate control terms 
            p_term = self.kp * error
            i_term = self.ki * error * period
            d_term = self.kd * (error - self.prev_error) / period
            # Update internal state
            self.prev_error = error
            self.integral += i_term
            # Get output
            out = p_term + self.integral + d_term
        else:
            out = 0.0
        return out


class SpatialController:

    def __init__(self) -> None:
        self.ctrl_l: list[Controller] = []

    def reset(self) -> None:
        """ Reset all sub-controllers """
        for ctrl in self.ctrl_l:
            ctrl.reset()

    def update(self, errors: npt.ArrayLike, period: float) -> npt.NDArray[np.float_]:
        """ Update sub-controllers

        Args:
            errors: Controller error list [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
            period: Controller update duration [sec]

        Returns:
            List with new controller outputs
        """
        out = [ctrl.update(err, period) for ctrl, err in zip(self.ctrl_l, np.reshape(errors, 6).tolist())]
        return np.array(out)


class SpatialPDController(SpatialController):

    def __init__(self, Kp_6: Sequence[float], Kd_6: Sequence[float]) -> None:
        """ Six dimensional spatial PD controller

        Args:
            Kp_6: 6 kp values
            Kd_6: 6 kd values
        """
        super().__init__()
        if len(Kp_6) == 6 and len(Kd_6) == 6:
            self.ctrl_l = [PDController(kp, kd) for kp, kd in zip(Kp_6, Kd_6)]
        else:
            raise ValueError(f"Given parameter lists '{Kp_6}' and '{Kd_6}' is not valid")


class SpatialPIDController(SpatialController):

    def __init__(self, Kp_6: Sequence[float], Ki_6: Sequence[float], Kd_6: Sequence[float]) -> None:
        """ Six dimensional spatial PID controller

        Args:
            Kp_6: 6 kp values
            Ki_6: 6 ki values
            Kd_6: 6 kd values
        """
        super().__init__()
        if len(Kp_6) == 6 and len(Ki_6) == 6 and len(Kd_6) == 6:
            self.ctrl_l = [PIDController(kp, ki, kd) for kp, ki, kd in zip(Kp_6, Ki_6, Kd_6)]
        else:
            raise ValueError(f"Given parameter lists '{Kp_6}', '{Ki_6}', and '{Kd_6}' is not valid")
