""" 
Script to move the robot arm randomly around the home position.
"""
# local
from ur_pilot import URPilot, move_joints_random


def move_joints_rnd() -> None:
    # Connect to robot arm
    ur10 = URPilot()

    # Move arm
    res = move_joints_random(ur10)
    # Print result
    print("Target joint positions: ", " ".join(f"{q:.3f}" for q in res.joint_pos))

    # Disconnect to robot arm
    ur10.exit()


if __name__ == '__main__':
    move_joints_rnd()
