from __future__ import annotations

# global
import logging
import numpy as np
from matplotlib import pyplot as plt

# local
from ur_pilot.utils import set_logging_level
from ur_pilot.config_mdl import FTSensor
from ur_pilot.end_effector import bota_helper as helper

# typing
from numpy import typing as npt


DURATION_ = 3.0


def plot6d(ts: npt.NDArray[np.float64], data_array: npt.NDArray[np.float64], x_label: str, y_labels: list[str]) -> None:
    fig, axs = plt.subplots(nrows=6)
    for i in range(6):
        axs[i].plot(ts, data_array[i])
        axs[i].set_ylabel(y_labels[i])
        axs[i].set_xlim(0, ts[-1])
        axs[i].grid(True)
    axs[-1].set_xlabel(x_label)
    fig.tight_layout()
    fig.align_labels()
    fig.subplots_adjust(hspace=0.3)


def main() -> None:

    cfg = FTSensor()
    cfg.adapter = "enp0s31f6"
    cfg.filter_fir = False
    cfg.filter_fast = False
    cfg.filter_chop = False
    cfg.filter_sinc_length = 51

    data = helper.record_from_sensor(cfg, DURATION_, 'ft_imu')
    t_span = np.linspace(0.0, DURATION_, data.shape[-1])
    ft_data = data[0:6]
    imu_data = data[6:12]

    # Plot force torque data
    x_label = 'time [s]'
    axs_labels = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Tx [Nm]', 'Ty [Nm]', 'Tz [Nm]']
    plot6d(t_span, ft_data, x_label, axs_labels)

    # Plot imu data
    axs_labels = ['Ax [g]', 'Ay [g]', 'Az [g]', 'Rx [rad/s]', 'Ry [rad/s]', 'Rz [rad/s]']
    plot6d(t_span, imu_data, x_label, axs_labels)
    plt.show()


if __name__ == '__main__':
    print("Bota force-torque sensor plot")
    set_logging_level(logging.INFO)
    main()
