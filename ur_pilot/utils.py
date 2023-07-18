""" This file contains common functionality. """
from __future__ import annotations

# global
import abc
import sys
import pysoem
import logging
from rigmopy import Vector6d

# typing
from typing import Sequence


def set_logging_level(level: int) -> None:
    """ Helper to configure logging

    Args:
        level: Logging level

    Returns:
        None
    """
    logging.basicConfig(format='%(levelname)s: %(message)s', level=level)


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

    def update(self, errors: Vector6d, period: float) -> Vector6d:
        """ Update sub-controllers

        Args:
            errors: Controller error list [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
            period: Controller update duration [sec]

        Returns:
            List with new controller outputs
        """
        out = [ctrl.update(err, period) for ctrl, err in zip(self.ctrl_l, errors.xyzXYZ)]
        return Vector6d().from_xyzXYZ(out)


class SpatialPDController(SpatialController):

    def __init__(self, Kp_6: Sequence[float], Kd_6: Sequence[float]) -> None:
        """ Six dimensional spatial PD controller

        Args:
            kp6: 6 kp values
            kd6: 6 kd values
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
