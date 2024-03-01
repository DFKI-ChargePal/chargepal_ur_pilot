""" Script for testing the plugging process combined with splitting the end-effector and the plug. """

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
_T_fpi2junction = sm.SE3().Trans([0.0, 0.0, -0.034 + 0.01])


def plugging_and_split(opt: Namespace) -> None:
    """ Function to run through the procedure

    Args:
        opt: Script arguments
    """
    # Connect to pilot
    with ur_pilot.connect() as pilot:
        # Apply transformation chain
        T_base2save_pre = _T_base2fpi * _T_fpi2save_pre
        T_base2junction = _T_base2fpi * _T_fpi2junction
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
            pilot.plug_in_force_ramp(f_axis='z', f_start=50.0, f_end=100, duration=3.0)
            # Check if robot is in target area
            xyz_base2fpi_base_est = _T_base2fpi.t
            xyz_base2fpi_base_meas = pilot.robot.tcp_pos
            error = np.linalg.norm(xyz_base2fpi_base_est - xyz_base2fpi_base_meas)
            if error > 0.01:
                print(f"The remaining position error {error} is quite large!")
            pilot.relax(2.0)
            # Release plug via twisting end-effector
            success = pilot.screw_ee_force_mode(4.0, -np.pi/2, 12.0)
            if not success:
                raise RuntimeError(f"Robot did not succeed in opening the twist lock. "
                                   f"Robot is probably in an undefined condition.")
            release_ft = np.array([0.0, 0.0, -25.0, 0.0, 0.0, 0.0])
            success = pilot.frame_force_mode(
                wrench=release_ft,
                compliant_axes=[1, 1, 1, 0, 0, 0],
                distance=0.04,
                time_out=5.0,
                frame='flange'
            )
            if not success:
                raise RuntimeError(
                    f"Error while trying to release the lock. Robot end-effector is probably still connected.")

        # End at retreat position
        time.sleep(2.0)
        with pilot.context.position_control():
            pilot.robot.movej([3.6226, -1.6558, 1.8276, 0.1048, 2.0615, -1.4946])


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
    plugging_and_split(args)
