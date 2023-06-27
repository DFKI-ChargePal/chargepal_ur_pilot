from __future__ import annotations

# global
import logging
import numpy as np
from matplotlib import pyplot as plt

# local
from ur_pilot.utils import set_logging_level
from ur_pilot.config_mdl import FTSensor
from ur_pilot import bota_helper as helper


DURATION_ = 3.0


def main() -> None:

    cfg1 = FTSensor()
    cfg1.filter_fir = False
    cfg1.filter_sinc_length = 128

    cfg2 = FTSensor()
    cfg2.filter_fir = True
    cfg2.filter_sinc_length = 128
    
    ft_signal1 = helper.record_from_sensor(cfg1, DURATION_, 'ft')[:, 10:]
    ft_signal2 = helper.record_from_sensor(cfg2, DURATION_, 'ft')[:, 10:]

    fig, axs = plt.subplots(nrows=6)
    t_span1 = np.linspace(0.0, DURATION_, ft_signal1.shape[-1])
    t_span2 = np.linspace(0.0, DURATION_, ft_signal2.shape[-1])
    if t_span1.shape[0] > t_span2.shape[0]:
        t_span = t_span2
    else:
        t_span = t_span1
    for i in range(6):
        axs[i].plot(t_span, ft_signal1[i, 0:t_span.shape[0]])
        axs[i].plot(t_span, ft_signal2[i, 0:t_span.shape[0]])
        axs[i].set_xlim(0, DURATION_)
        axs[i].grid(True)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    plt.show()


if __name__ == '__main__':
    print("Bota force-torque sensor plot")
    set_logging_level(logging.INFO)
    main()
