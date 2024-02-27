"""
Script to show the motion mode with fast target updates
"""
# libs
import time
import ur_pilot
import numpy as np
import spatialmath as sm

_episode_len = 20.0  # [sec]
_control_freq = 20  # [hz]


def motion_mode_ctrl(episode_len: float) -> None:
    # Connect to pilot/robot arm
    with ur_pilot.connect() as pilot:
        # Move home
        with pilot.context.position_control():
            pilot.robot.move_home()

        with pilot.context.motion_control():
            ref_pose = pilot.robot.tcp_pose
            ref_pos, ref_rot = ref_pose.t, ref_pose.R
            t_start_ = time.time()
            dt = 0
            while time.time() - t_start_ < episode_len:
                dt += 1
                new_pose = sm.SE3.Rt(
                    R=ref_rot * sm.SO3.RPY(0.005 * np.random.randn(3)),
                    t=ref_pos + 0.005 * np.random.randn(3)
                )
                t_sub_start = time.time()
                while time.time() - t_sub_start < 1/_control_freq:
                    pilot.robot.motion_mode(new_pose)


if __name__ == '__main__':
    motion_mode_ctrl(_episode_len)
