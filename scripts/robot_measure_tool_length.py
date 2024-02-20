""" Script to measure the tool length in a certain orientation """
import time
import logging
import argparse
import ur_pilot
import numpy as np
from rigmopy import Pose, Vector3d, Quaternion

LOGGER = logging.getLogger(__name__)

rot_15_deg = np.pi / 12

START_JOINT_POS = [3.193, -1.099, 1.928, 1.003, 1.555, -1.520]
START_POSE_STRAIGHT = Pose().from_xyz([0.469, 0.202, 0.077]).from_axis_angle([0.0, - np.pi, 0.0])
START_POSE_TILTED = Pose().from_xyz([0.469, 0.202, 0.077]).from_axis_angle([0.0, - np.pi + rot_15_deg, 0.0])

# tilted_offset = Vector3d(-0.025, 0.0, 0.15122), (0.0, -0.2618, 0.0)


def main() -> None:

    # Connect to pilot
    with ur_pilot.connect() as pilot:

        # Move to home position
        with pilot.context.position_control():
            # Move to home position
            pilot.move_home()
            # Move to tiled position
            pilot.move_to_tcp_pose(START_POSE_TILTED)

        # Measure the surface with tilted tool
        with pilot.context.force_control():
            success, contact_flange_pose_tilted = pilot.find_contact_point(direction=[0, 0, -1, 0, 0, 0], time_out=5.0)
            time.sleep(1.0)
            if success:
                LOGGER.info(f"Find contact pose: {contact_flange_pose_tilted}")
            success_retreat, _ = pilot.retreat(6*[0.0], [0, 0, 1, 0, 0, 0], distance=0.05, time_out=5.0)

        if not success_retreat:
            raise RuntimeError("Moving to retreat pose was not successful. Stop moving.")

        with pilot.context.position_control():
            # Move to straight position
            pilot.move_to_tcp_pose(START_POSE_STRAIGHT)

        # Measure the surface with straight tool
        with pilot.context.force_control():
            success, contact_flange_pose_straight = pilot.find_contact_point(direction=[0, 0, -1, 0, 0, 0], time_out=5.0)
            time.sleep(1.0)
            if success:
                LOGGER.info(f"Find contact pose: {contact_flange_pose_straight}")
            success_retreat, _ = pilot.retreat(6*[0.0], [0, 0, 1, 0, 0, 0], distance=0.05, time_out=5.0)

        if not success_retreat:
            raise RuntimeError("Moving to retreat pose was not successful. Stop moving.")

        with pilot.context.position_control():
            # Move back to home position()
            pilot.move_home()

        # Calculate tilted tool offset.
        tcp_offset_straight = pilot.robot.tcp_offset
        # We are only interested in Z - direction
        z_flange_tilted = contact_flange_pose_tilted.xyz[-1]
        z_flange_straight = contact_flange_pose_straight.xyz[-1]
        z_diff = z_flange_straight - z_flange_tilted
        tcp_offset_tilted = Pose().from_pq(p=tcp_offset_straight.p - Vector3d().from_xyz([0.025, 0.0, z_diff]))
        tcp_offset_tilted = tcp_offset_tilted.from_pq(q=tcp_offset_straight.q * Quaternion().from_axis_angle([0.0, -rot_15_deg, 0.0])) 

        LOGGER.info(f"Find new offset for tilted tool with pose: {tcp_offset_tilted.p, tcp_offset_tilted.axis_angle}")


if __name__ == '__main__':
    """ Script to measure the tool length in a odd orientation """
    parser = argparse.ArgumentParser(description="Demo to measure the tool length in a odd orientation.")
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        ur_pilot.logger.set_logging_level(logging.DEBUG)
    else:
        ur_pilot.logger.set_logging_level(logging.INFO)
    main()
