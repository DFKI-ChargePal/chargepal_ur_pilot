import time
import ur_pilot
import numpy as np


# _jp1 = np.array([3.132, -1.371, 2.127, -0.746, 1.602, -1.565])
# _jp2 = np.array([3.269, -1.392, 1.901, -0.250, 1.657, -1.565])
# _jp3 = np.array([3.418, -1.282, 1.939, -0.598, 1.827, -1.565])
# _jp4 = np.array([3.362, -1.211, 2.001, -0.782, 1.798, -1.565])

_jp1 = np.array([-3.1329, -2.0758, 2.3816, -0.3084, 1.5888, -1.4787])
_jp2 = np.array([-3.1321, -2.0757, 2.4131, 0.3912, 1.0184, -1.4786])
_jp3 = np.array([-3.1321, -2.0758, 2.4153, 1.4929, 0.1311, -1.4787])
_jp4 = np.array([-3.1340, -2.0468, 2.4155, 2.4546, -0.9318, -1.4788])
_jp5 = np.array([-3.1337, -2.0469, 2.4808, 3.2476, -1.6301, -1.4788])
_jp6 = np.array([-3.1338, -1.8509, 1.8928, 3.2475, -1.6301, -1.4788])
_jp7 = np.array([-3.1338, -1.0364, -2.2332, 3.2469, -1.6301, -1.4789])
_jp8 = np.array([0.0361, -1.0378, -2.2332, 3.2469, -1.6302, -1.5099])

wps_fwd = [_jp1, _jp2, _jp3, _jp4, _jp5, _jp6, _jp7, _jp8]
wps_bwd = wps_fwd[::-1]


def move_joint_path() -> None:

    # Connect to pilot/robot arm
    with ur_pilot.connect() as pilot:
        # Move home
        with pilot.context.position_control():
            # Move along the path waypoints
            velocity = 0.5
            acceleration = 0.1
            pilot.robot.move_path_j(wps_fwd, velocity, acceleration)
            time.sleep(3.0)
            pilot.robot.move_path_j(wps_bwd, velocity, acceleration)


if __name__ == "__main__":
    move_joint_path()
