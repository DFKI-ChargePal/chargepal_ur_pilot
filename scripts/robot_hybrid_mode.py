""" Script to show the hybrid mode in action
"""
# global
import logging
import argparse
import numpy as np
import spatialmath as sm
from time import perf_counter

# local
import ur_pilot

LOGGER = logging.getLogger(__name__)


def hybrid_mode_ctrl() -> None:
    # Connect to pilot/robot arm
    with ur_pilot.connect() as pilot:
        # Move home
        with pilot.context.position_control():
            pilot.robot.move_home()

        # Get current pose as reference
        ref_pose = pilot.robot.tcp_pose
        ref_pos, ref_rot = ref_pose.t, ref_pose.R
        with pilot.context.hybrid_control():
            # Shift pose randomly relative to the reference pose
            new_pose = sm.SE3.Rt(
                R=ref_rot * sm.SO3.RPY(0.025 * np.random.randn(3)),
                t=ref_pos + 0.005 * np.random.randn(3)
            )
            for i in range(50):
                t_start_ = perf_counter()
                wrench = sm.SpatialForce(0.1 * np.random.randn(6))
                while perf_counter() - t_start_ < 1/5:  # run for xx seconds
                    pilot.robot.hybrid_mode(new_pose, wrench)


if __name__ == '__main__':
    """ Script to show the hybrid mode in action """
    parser = argparse.ArgumentParser(description="Show robots hybrid control mode")
    parser.add_argument('--debug', action='store_true', help='Print additional debug information')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        ur_pilot.logger.set_logging_level(logging.DEBUG)
    else:
        ur_pilot.logger.set_logging_level(logging.INFO)
    # Run demo
    hybrid_mode_ctrl()
