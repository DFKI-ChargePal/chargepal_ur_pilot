import numpy as np

import ur_pilot

_jp1 = np.array([3.132, -1.371, 2.127, -0.746, 1.602, -1.565])
_jp2 = np.array([3.269, -1.392, 1.901, -0.250, 1.657, -1.565])
_jp3 = np.array([3.418, -1.282, 1.939, -0.598, 1.827, -1.565])
_jp4 = np.array([3.362, -1.211, 2.001, -0.782, 1.798, -1.565])


def move_joint_path() -> None:

    # Connect to pilot/robot arm
    with ur_pilot.connect() as pilot:
        # Move home
        with pilot.position_control():
            pilot.move_home()
            # Move along the path waypoints
            velocity = 0.1
            acceleration = 0.1
            wps = [_jp1, _jp2, _jp3, _jp4]
            pilot.robot.movej_path(wps, velocity, acceleration)


if __name__ == "__main__":
    move_joint_path()
