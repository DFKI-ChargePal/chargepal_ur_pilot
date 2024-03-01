""" Script for testing the pure plugging process. Means inserting and removing the plug in one go. """

# global
import time
import logging
import argparse
import ur_pilot
import numpy as np
import spatialmath as sm

# typing
from argparse import Namespace

# Constants
_T_base2fpi = sm.SE3.Rt(R=sm.SO3.EulerVec((0.005, 1.568, -0.010)), t=(0.935, 0.294, 0.477))
_T_fpi2save_pre = sm.SE3().Trans([0.0, 0.0, -0.034 - 0.02])
_T_fpi2junction = sm.SE3().Trans([0.0, 0.0, -0.034 + 0.02])


def plugging(opt: Namespace) -> None:
    """ Function to run through the plugging procedure

    Args:
        opt: Script arguments
    """
    # Connect to pilot
    with ur_pilot.connect() as pilot:
        # Apply transformation chain
        T_base2save_pre = _T_base2fpi * _T_fpi2save_pre
        T_base2junction = _T_base2fpi @ _T_fpi2junction
        # Free space movements
        with pilot.context.position_control():
            # Start at home position
            pilot.robot.move_home()
            # Move to pre-connect pose
            pilot.move_to_tcp_pose(T_base2save_pre)
        time.sleep(2.0)

        # Getting in junction between plug and socket
        with pilot.context.motion_control():
            pilot.move_to_tcp_pose(T_base2junction, time_out=4.0)
            # Check if robot is in target area
            xyz_base2jct_base_est = T_base2junction.t
            xyz_base2jct_base_meas = pilot.robot.tcp_pos
            error = np.linalg.norm(xyz_base2jct_base_est - xyz_base2jct_base_meas)
            if error > 0.01:
                raise RuntimeError(f"Remaining position error {error} to alignment state is to large. "
                                f"Robot is probably in an undefined condition.")
        # Start to apply some force
        with pilot.context.force_control():
            # Try to fully plug in
            pilot.plug_in_force_ramp(f_axis='z', f_start=70.0, f_end=110, duration=4.0)
            # Check if robot is in target area
            xyz_base2fpi_base_est = _T_base2fpi.t
            xyz_base2fpi_base_meas = pilot.robot.tcp_pos
            error = np.linalg.norm(xyz_base2fpi_base_est - xyz_base2fpi_base_meas)
            if error > 0.01:
                print(f"The remaining position error {error} is quite large!")
            pilot.relax(2.0)
            # Plug out again
            plug_out_ft = np.array([0.0, 0.0, -100.0, 0.0, 0.0, 0.0])
            success = pilot.tcp_force_mode(
                wrench=plug_out_ft,
                compliant_axes=[0, 0, 1, 0, 0, 0],
                distance=0.06,  # 6cm
                time_out=10.0)
            if not success:
                raise RuntimeError(f"Error while trying to unplug. Plug is probably still connected.")

        # End at home position
        time.sleep(2.0)
        with pilot.context.position_control():
            pilot.robot.move_home()


if __name__ == '__main__':
    """ Entry point to start process. """
    # Input parsing
    parser = argparse.ArgumentParser(description='Script to connect the plug with the socket.')
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    args = parser.parse_args()
    if args.debug:
        args.logging_level = logging.DEBUG
    else:
        args.logging_level = logging.INFO
    plugging(args)
