""" Script to show the hybrid mode in action
"""
# global
import logging
import argparse
import numpy as np
from time import perf_counter
from rigmopy import Vector6d
from chargepal_aruco import set_logging_level

# local
import ur_pilot

LOGGER = logging.getLogger(__name__)


def hybrid_mode_ctrl() -> None:
    # Connect to pilot/robot arm
    with ur_pilot.connect() as pilot:
        # Move home
        with pilot.position_control():
            pilot.move_home()

        # Get current pose as reference
        ref_pose = pilot.robot.get_tcp_pose()
        with pilot.hybrid_control():

            for i in range(6):
                # Shift pose randomly relative to the reference pose
                rel_pose = ref_pose.random((0.05, 0.05, 0.05), (3*180/np.pi, 3*180/np.pi, 3*180/np.pi))

                t_start_ = perf_counter()
                while perf_counter() - t_start_ < 3.0:  # run for 3 seconds
                    pilot.robot.hybrid_mode(rel_pose, Vector6d())


if __name__ == '__main__':
    """ Script to show the hybrid mode in action """
    parser = argparse.ArgumentParser(description="Show robots hybrid control mode")
    parser.add_argument('--debug', action='store_true', help='Print additional debug information')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        set_logging_level(logging.DEBUG)
    else:
        set_logging_level(logging.INFO)
    # Run demo
    hybrid_mode_ctrl()
