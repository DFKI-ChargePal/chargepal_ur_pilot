# global
import numpy as np
from dataclasses import dataclass
# typing
from numpy import typing as npt


@dataclass
class CameraCoefficient:
    """ Class to represent camera coefficients

    Args:
        intrinsic: Intrinsic camera matrix for the raw (distorted) images
        distortion: The distortion parameters
    """
    intrinsic: npt.NDArray[np.float64] = np.identity(3)
    distortion: npt.NDArray[np.float64] = np.zeros(4)
