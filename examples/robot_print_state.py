""" 
Script to print the current robot state.
Contains: Joint positions
          TCP pose
"""
# local
import ur_pilot


if __name__ == '__main__':
    # Connect to API
    with ur_pilot.connect() as pilot:
            # Read joint and pose information
        joint_pos = pilot.robot.get_joint_pos()
        tcp_pose = pilot.robot.get_tcp_pose()
        # Print out
        print("      Joint positions: ", " ".join(f"{q:.3f}" for q in joint_pos))
        print("TCP pose w. axis ang.: ", " ".join(f"{p:.3f}" for p in tcp_pose.xyz + tcp_pose.axis_angle))
