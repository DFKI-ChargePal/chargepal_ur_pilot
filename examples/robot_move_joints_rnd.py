""" Script to move the robot arm randomly around the home position """
import ur_pilot


if __name__ == '__main__':
    with ur_pilot.connect() as pilot:
        with pilot.position_control():
            rnd_joint_pos = pilot.move_joints_random()
            # Print result
            print("Target joint positions: ", " ".join(f"{q:.3f}" for q in rnd_joint_pos))
