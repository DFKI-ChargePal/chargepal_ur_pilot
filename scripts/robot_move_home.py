""" Script to move the robot arm to the home position """
import ur_pilot
from config import config_data


if __name__ == '__main__':
    with ur_pilot.connect(config_data.robot_dir) as pilot:
        with pilot.context.position_control():
            pilot.robot.move_home()
