""" Short demo to demonstrate motion mode. Goal is to keep the current pose """
import time
from ur_pilot import URPilot


def keep_pose(soft: bool, time_out: float) -> None:

    # Connect to robot arm
    ur10 = URPilot()

    if soft:
        ur10.set_up_motion_mode(Kp_6=[1.0, 1.0, 1.0, 0.01, 0.01, 0.01])
    else:
        ur10.set_up_motion_mode(Kp_6=[25.0, 25.0, 25.0, 1.0, 1.0, 1.0])
    init_pose = ur10.get_tcp_pose()
    t_start = time.time()

    try:

        while time.time() - t_start < time_out:
            ur10.motion_mode(init_pose)

    except Exception as e:
        print(e)
    finally:
        ur10.stop_motion_mode()
        ur10.exit()


if __name__ == '__main__':
    keep_pose(True, 30.0)
