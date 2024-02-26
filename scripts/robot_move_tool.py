"""
Script to move w.r.t tool frame
"""
# libs
import logging
import ur_pilot
import numpy as np
import spatialmath as sm


LOGGER = logging.getLogger(__name__)


def move_tool() -> None:
    ur_pilot.logger.set_logging_level(logging.INFO)
    # Connect to pilot/robot
    with ur_pilot.connect() as pilot:
        # Move home
        with pilot.context.position_control():
            pilot.robot.move_home()
            # Align tool to table
            T_base2tcp = pilot.robot.tcp_pose
            LOGGER.info(f"Starting pose: {ur_pilot.utils.se3_to_str(T_base2tcp)}")
            T_base2target = sm.SE3.Rt(R=sm.SO3.EulerVec([0.0, np.pi/2, 0.0]), t=pilot.robot.tcp_pos.tolist())
            LOGGER.info(f"Finale pose  : {ur_pilot.utils.se3_to_str(T_base2target)}")
            pilot.move_to_tcp_pose(T_base2target)


if __name__ == '__main__':
    move_tool()
