# global
import time
import logging
import argparse
import numpy as np

# local
from ur_pilot.core import URPilot
from ur_pilot.utils import set_logging_level
from ur_pilot.monitor.signal_monitor import SignalMonitor


LOGGER = logging.getLogger(__name__)

TEACH_IN_FILE_ = "ft_sensor_calibration.json"
AX_LABELS_ = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Tx [Nm]', 'Ty [Nm]', 'Tz [Nm]']


def follow_state_sequence() -> None:

    # Connect to robot
    ur10 = URPilot()
    ur10.move_home()

    # Create a monitor to display ft readings
    ft_monitor = SignalMonitor(AX_LABELS_, round(1 / ur10.ft_sensor.time_step), 10.0)
    sub_steps = int(ft_monitor.display_rate / ur10.ft_sensor.time_step)
    LOGGER.info(f'Perform {sub_steps} sub-steps before updating the Monitor.')

    # Enable free drive mode
    ur10.teach_mode()
    LOGGER.info("Start free-drive mode: ")
    try:
        t_start_ = time.time()
        t_next_ = t_start_
        while True:
            ft_signal = np.reshape(ur10.ft_sensor.FT, [6, 1])
            for i in range(sub_steps - 1):
                ft_signal = np.hstack([ft_signal, np.reshape(ur10.ft_sensor.FT, [6, 1])])
                time.sleep(ur10.ft_sensor.time_step)

            t_next_ += ft_monitor.display_rate
            t_left_ = t_next_ - time.time()
            if t_left_ > 0.0:
                time.sleep(t_left_)
            ft_monitor.add(ft_signal)

    finally:
        # Stop free drive mode
        ur10.stop_teach_mode()
        # Clean up
        ur10.exit()


if __name__ == '__main__':
    """ Script to move the robot by hand and display force-torque readings in parallel """
    parser = argparse.ArgumentParser(description="Demo to demonstrate FT-sensor calibration results.")
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        set_logging_level(logging.DEBUG)
    else:
        set_logging_level(logging.INFO)

    follow_state_sequence()
