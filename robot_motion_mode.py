"""
Script to show the motion mode with fast target updates
"""
# global
import time
import numpy as np
from rigmopy import Pose, Quaternion

# local
from ur_pilot import URPilot

EPISODE_LEN_ = 20.0  # [sec]
CONTROL_FREQ_ = 20  # [hz]



def motion_mode_ctrl(episode_len: float) -> None:
    # Connect to robot arm
    ur10 = URPilot()
    try:
        # Move home
        ur10.move_home()
        ur10.set_up_motion_mode()
        ref_pose = ur10.get_tcp_pose()
        ref_p, ref_q = ref_pose.p, ref_pose.q
        t_start_ = time.time()
        dt = 0
        while time.time() - t_start_ < episode_len:
            dt += 1
            new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + 0.01 * np.random.randn(3), (ref_q * Quaternion().from_axis_angle(0.001 * np.random.randn(3))).wxyz)
            # new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + np.array([0.0, - dt * 0.02, 0.0]), ref_q.wxyz)
            # new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz), ref_q.wxyz)
            t_sub_start = time.time()
            while time.time() - t_sub_start < 1/CONTROL_FREQ_:
                ur10.motion_mode(new_pose)
    except Exception as e:
        print(e)
    finally:
        # Clean up
        ur10.stop_motion_mode()
        ur10.exit()


if __name__ == '__main__':
    motion_mode_ctrl(EPISODE_LEN_)
