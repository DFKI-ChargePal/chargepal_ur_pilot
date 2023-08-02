""" Script to measure the tool length in a certain orientation """

import time
import ur_pilot
import numpy as np
from rigmopy import Pose, Vector6d

rot_15_deg = np.pi / 12


START_JOINT_POS = [3.193, -1.099, 1.928, 1.003, 1.555, -1.520]
START_POSE = Pose().from_xyz([0.469, 0.202, 0.077]).from_axis_angle([0.0, - np.pi + rot_15_deg,0.0])


def main() -> None:

    # Connect to pilot
    with ur_pilot.connect() as pilot:

        # Move to home position
        with pilot.position_control():
            pilot.move_home()
            pilot.move_to_tcp_pose(START_POSE)

        with pilot.force_control():
            pilot.find_contact_point(direction=[0, 0, -1, 0, 0, 0], time_out=5)
            time.sleep(3)
            pilot.plug_out_force_mode(
                wrench=Vector6d().from_xyzXYZ([0.0, 0.0, -10.0, 0.0, 0.0, 0.0]),
                compliant_axes=[0, 0, 1, 0, 0, 0],
                distance=0.05,
                time_out=5.0)


if __name__ == '__main__':
    main()
