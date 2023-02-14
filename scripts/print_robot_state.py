""" 
Script to print the the current robot state. 
Contains: Joint positions
          TCP pose
"""
# local
from ur_pilot.core.robot import Robot


def print_robot_state() -> None:
    # Connect to robot arm
    ur10 = Robot()

    # Read joint and pose information
    joint_pos = ur10.get_joint_pos()
    tcp_pose = ur10.get_tcp_pose()

    # Print out
    print("Joint Positions: ", " ".join(f"{q:.3f}" for q in joint_pos))
    print("       TCP Pose: ", " ".join(f"{p:.3f}" for p in tcp_pose))

    # Disconnect to robot arm
    ur10.exit()


if __name__ == '__main__':
    print_robot_state()
