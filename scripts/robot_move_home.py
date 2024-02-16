""" Script to move the robot arm to the home position """
import ur_pilot


if __name__ == '__main__':
    with ur_pilot.connect() as pilot:
        with pilot.context.position_control():
            pilot.move_home()
