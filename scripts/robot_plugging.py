""" Script for testing the pure plugging process. Means inserting and removing the plug in one go. """

# global
import time
import logging
import argparse
import ur_pilot
import numpy as np
from rigmopy import Pose, Vector6d

# typing
from argparse import Namespace

# Constants
_pose_base2fpi = Pose().from_xyz((0.935, 0.294, 0.477)).from_axis_angle((0.005, 1.568, -0.010))
_pose_fpi2save_pre = Pose().from_xyz([0.0, 0.0, -0.034 - 0.02])
_pose_fpi2junction = Pose().from_xyz(xyz=[0.0, 0.0, -0.034 + 0.01])
# _pose_socket2fpi = Pose().from_xyz(xyz=[0.0, 0.0, 0.034])


def plugging(opt: Namespace) -> None:
    """ Function to run through the plugging procedure

    Args:
        opt: Script arguments
    """

    # Connect to pilot
    with ur_pilot.connect() as pilot:
        # Get transformation matrices
        T_base2fpi = _pose_base2fpi.transformation
        T_fpi2save_pre = _pose_fpi2save_pre.transformation
        T_fpi2junction = _pose_fpi2junction.transformation
        # Apply transformation chain
        T_base2save_pre = T_base2fpi @ T_fpi2save_pre
        T_base2junction = T_base2fpi @ T_fpi2junction
        # Free space movements
        with pilot.position_control():
            # Start at home position
            pilot.move_home()
            # Move to pre-connect pose
            pilot.move_to_tcp_pose(T_base2save_pre.pose)
        time.sleep(2.0)

        # Getting in junction between plug and socket
        with pilot.motion_control():
            pilot.move_to_tcp_pose(T_base2junction.pose, time_out=3.0)
        # Check if robot is in target area
        xyz_base2jct_base_est = np.reshape(T_base2junction.tau, 3)
        xyz_base2jct_base_meas = np.reshape(pilot.robot.get_tcp_pose().xyz, 3)
        error = np.sqrt(np.sum(np.square(xyz_base2jct_base_est - xyz_base2jct_base_meas)))
        if error > 0.01:
            raise RuntimeError(f"Remaining position error {error} to alignment state is to large. "
                                f"Robot is probably in an undefined condition.")
        # Start to apply some force
        time.sleep(2.0)
        with pilot.force_control():
            # Try to fully plug in
            pilot.plug_in_force_ramp(f_axis='z', f_start=50.0, f_end=90, duration=3.0)
            # Check if robot is in target area
            xyz_base2fpi_base_est = np.reshape(T_base2fpi.tau, 3)
            xyz_base2fpi_base_meas = np.reshape(pilot.robot.get_tcp_pose().xyz, 3)
            error = np.sqrt(np.sum(np.square(xyz_base2fpi_base_est - xyz_base2fpi_base_meas)))
            if error > 0.01:
                print(f"Remaining position error {error} quite large!")
            pilot.relax(2.0)
            # Plug out again
            plug_out_ft = Vector6d().from_xyzXYZ([0.0, 0.0, -100.0, 0.0, 0.0, 0.0])
            success = pilot.tcp_force_mode(
                wrench=plug_out_ft,
                compliant_axes=[0, 0, 1, 0, 0, 0],
                distance=0.060,
                time_out=10.0)
            if not success:
                raise RuntimeError(f"Error while trying to unplug. Plug is probably still connected.")

        # End at home position
        time.sleep(2.0)
        with pilot.position_control():
            pilot.move_home()


def main() -> None:
    """ Main function to start process. """
    # Input parsing
    parser = argparse.ArgumentParser(description='Script to connect the plug with the socket.')
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    args = parser.parse_args()
    if args.debug:
        args.logging_level = logging.DEBUG
    else:
        args.logging_level = logging.INFO
    plugging(args)


if __name__ == '__main__':
    main()
