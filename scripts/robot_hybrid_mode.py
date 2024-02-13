""" Script to show the hybrid mode in action
"""
# global
import logging
import argparse
import numpy as np
from time import perf_counter
from rigmopy import Vector6d
import spatialmath as sm

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
            r = sm.UnitQuaternion(new_pose.q.wxyz).SO3()
            target_se3 = sm.SE3.Rt(R=r, t=new_pose.p.xyz)

            for i in range(50):
                t_start_ = perf_counter()
                wrench = np.random.randn(1)[-1] * Vector6d().from_random()

                wrench = sm.SpatialForce(wrench.xyzXYZ)
                # wrench = Vector6d().from_xyzXYZ([0, 0, 1, 0, 0, 0])
                while perf_counter() - t_start_ < 1/5:  # run for xx seconds
                    pilot.robot.hybrid_controller.update(target_se3, wrench)


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
