""" 
Script to move the robot arm randomly around the home position.
"""
# local
from ur_pilot.core.robot import Robot
from ur_pilot.core.move_j_rnd import move_joints_random


def move_joints_rnd() -> None:
    # Connect to robot arm
    ur10 = Robot()

    # Move arm
    res = move_joints_random(ur10)
    # Print result
    print("Target joint positions: ", " ".join(f"{q:.3f}" for q in res.rnd_joint_pos))

    # Disconnect to robot arm
    ur10.exit()


if __name__ == '__main__':
    move_joints_rnd()
