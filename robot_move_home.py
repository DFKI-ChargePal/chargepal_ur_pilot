""" 
Script to move the robot arm to the home position.
"""
# local
from ur_pilot import URPilot


def move_home() -> None:
    # Connect to robot arm
    ur10 = URPilot()
    tgt_j_pos = " ".join(f"{q:.3f}" for q in ur10.home_joint_config)
    print("Start moving to home configuration:" ,tgt_j_pos)
    # Move arm
    ur10.move_home()
    # Print result
    print("               New joint positions:", " ".join(f"{q:.3f}" for q in ur10.get_joint_pos()))

    # Disconnect to robot arm
    ur10.exit()


if __name__ == '__main__':
    move_home()
