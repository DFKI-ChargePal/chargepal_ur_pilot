from __future__ import annotations

# global
import numpy as np
from rigmopy import Vector3d, Vector6d

# typing
from numpy import typing as npt


class FTCalibration:

    def __init__(self) -> None:
        """ Class based on the ROS package http://wiki.ros.org/force_torque_tools
        """
        self.num_meas = 0
        self.H: npt.NDArray[np.float64] | None = None  # stacked measurement matrices
        self.Z: npt.NDArray[np.float64] | None = None  # stacked F/T measurements

    def add_measurement(self, gravity: Vector3d, ft_raw: Vector6d) -> None:
        """ Add a new measurement for calibration. Make sure both vector are represented in the same frame.

        Args:
            gravity: The gravity vector w.r.t. the world frame
            ft_raw:  The force-torque readings w.r.t. the world frame

        Returns:
            None
        """
        h = self._get_measurement_matrix(gravity)
        z = np.reshape(ft_raw.to_numpy(), [6, 1])

        if self.num_meas <= 0:
            self.H = h
            self.Z = z
        else:
            self.H = np.vstack([self.H, h])
            self.Z = np.vstack([self.Z, z])

        self.num_meas += 1

    def get_calib(self) -> tuple[float, Vector3d, Vector6d]:
        """ Least squares to estimate the FT sensor parameters

        Returns:
            Calibration parameters [mass, center of mass, ft bias]
        """
        ft_calib_params = np.linalg.lstsq(self.H, self.Z, rcond=None)[0]

        mass = ft_calib_params[0]
        if mass <= 0.0:
            raise ValueError(f"Error in estimated mass '{mass}' (<= 0.0)")
        com = Vector3d().from_xyz([
            ft_calib_params[1] / mass,
            ft_calib_params[2] / mass,
            ft_calib_params[3] / mass
            ])
        ft_bias = Vector6d().from_xyzXYZ([
            -ft_calib_params[4],
            -ft_calib_params[5],
            -ft_calib_params[6],
            -ft_calib_params[7],
            -ft_calib_params[8],
            -ft_calib_params[9],
        ])
        return mass, com, ft_bias

    @staticmethod
    def _get_measurement_matrix(gravity: Vector3d) -> npt.NDArray[np.float64]:
        """ Measurement matrix based on >> On-line Rigid Object Recognition and Pose Estimation Based on
            Inertial Parameters , D. Kubus, T. Kroger, F. Wahl, IROS 2008 <<

        Args:
            gravity: The gravity vector w.r.t. the world frame

        Returns:
            A 6x10 measurement matrix
        """
        # Measurement matrix
        H = np.zeros([6, 10], dtype=np.float64)
        # We assume stationary measurements
        w = np.zeros([3, 1], dtype=np.float64)
        a = np.zeros([3, 1], dtype=np.float64)
        alpha = np.zeros([3, 1], dtype=np.float64)
        g = np.reshape(gravity.xyz, [3, 1])

        # Make upper right part of matrix a diagonal matrix
        for i in range(3):
            for j in range(4, 10):
                if i == j-4:
                    H[i, j] = 1.0
                else:
                    H[i, j] = 0.0

        # Set zeros at lower left matrix part
        for i in range(3, 6):
            H[i, 0] = 0.0
        H[3, 1] = 0.0
        H[4, 2] = 0.0
        H[5, 3] = 0.0

        # Fill upper left matrix part with values
        for i in range(3):
            H[i, 0] = a[i] - g[i]
        H[0, 1] = -w[1] * w[1] - w[2] * w[2]
        H[0, 2] = w[0] * w[1] - alpha[2]
        H[0, 3] = w[0] * w[2] + alpha[1]

        H[1, 1] = w[0] * w[1] + alpha[2]
        H[1, 2] = -w[0] * w[0] - w[2] * w[2]
        H[1, 3] = w[1] * w[2] - alpha[0]

        H[2, 1] = w[0] * w[2] - alpha[1]
        H[2, 2] = w[1] * w[2] + alpha[0]
        H[2, 3] = -w[1] * w[1] - w[0] * w[0]

        # -- Fill lower left matrix part with values
        H[3, 2] = a[2] - g[2]
        H[3, 3] = g[1] - a[1]

        H[4, 1] = g[2] - a[2]
        H[4, 3] = a[0] - g[0]

        H[5, 1] = a[1] - g[1]
        H[5, 2] = g[0] - a[0]

        # Make lower right part of matrix a diagonal matrix (no movement in measurements)
        for i in range(3, 6):
            for j in range(4, 10):
                if i == j-4:
                    H[i, j] = 1.0
                else:
                    H[i, j] = 0.0

        return H
