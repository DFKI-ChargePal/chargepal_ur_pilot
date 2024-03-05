""" Script to move the robot arm to the home position """
import ur_pilot
from config import data


if __name__ == '__main__':
    with ur_pilot.connect(data.robot_dir) as pilot:
        with pilot.context.position_control():
            pilot.robot.move_home()
