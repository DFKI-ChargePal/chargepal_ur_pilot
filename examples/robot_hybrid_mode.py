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

            # Shift pose randomly relative to the reference pose
            new_pose = ref_pose.random((0.025, 0.025, 0.025), (np.deg2rad(3), np.deg2rad(3), np.deg2rad(3))) 
            
            for i in range(50):
                t_start_ = perf_counter()
                wrench = np.random.randn(1)[-1] * Vector6d().from_random()
                # wrench = Vector6d().from_xyzXYZ([0, 0, 1, 0, 0, 0])
                while perf_counter() - t_start_ < 1/5:  # run for xx seconds
                    pilot.robot.hybrid_mode(new_pose, wrench)


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
