# global
import time
import logging
import argparse
import numpy as np

# local
import ur_pilot


LOGGER = logging.getLogger(__name__)

TEACH_IN_FILE_ = "ft_sensor_calibration.json"
AX_LABELS_ = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Tx [Nm]', 'Ty [Nm]', 'Tz [Nm]']


def follow_state_sequence() -> None:

    # Connect to robot
    with ur_pilot.connect() as pilot:

        with pilot.position_control():
            pilot.move_home()

        # Create a monitor to display ft readings
        ft_monitor = ur_pilot.SignalMonitor(AX_LABELS_, round(1 / pilot.robot.ft_sensor.time_step), 3.14)
        sub_steps = int(ft_monitor.display_rate / pilot.robot.ft_sensor.time_step)
        LOGGER.info(f'Perform {sub_steps} sub-steps before updating the Monitor.')

        # Enable free drive mode
        with pilot.teach_in_control():    
            LOGGER.info("Start free-drive mode: ")
            t_start_ = time.time()
            t_next_ = t_start_
            while True:
                ft_signal = np.reshape(pilot.robot.get_tcp_force(extern=True).xyzXYZ, [6, 1])
                for i in range(sub_steps - 1):
                    ft_signal = np.hstack([ft_signal, np.reshape(pilot.robot.get_tcp_force(extern=True).xyzXYZ, [6, 1])])
                    time.sleep(pilot.robot.ft_sensor.time_step)

                t_next_ += ft_monitor.display_rate
                t_left_ = t_next_ - time.time()
                if t_left_ > 0.0:
                    time.sleep(t_left_)
                ft_monitor.add(ft_signal)



if __name__ == '__main__':
    """ Script to move the robot by hand and display force-torque readings in parallel """
    parser = argparse.ArgumentParser(description="Demo to demonstrate FT-sensor calibration results.")
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        ur_pilot.set_logging_level(logging.DEBUG)
    else:
        ur_pilot.set_logging_level(logging.INFO)

    follow_state_sequence()
