# global
import time
import logging

# local
from ur_pilot.config_server import ConfigServer
from ur_pilot.rtde_interface import RTDEInterface

# typing
from typing import List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


class Robot:

    def __init__(self) -> None:
        # Robot interface
        self.rtde = RTDEInterface(verbose=True)
        # joint space movements
        self.joint_vel: float = 0.25
        self.joint_acc: float = 0.10
        # linear space movements
        self.lin_vel: float = 0.10
        self.lin_acc: float = 0.10
        # linear servoing
        self.lin_servo_vel: float = 0.0  # NOT USED
        self.lin_servo_acc: float = 0.0  # NOT USED
        self.lin_servo_gain: float = 100.0  # range: [100, 2000]
        self.lin_servo_lkh_time: float = 0.2  # range: [0.03, 0.2]
        # force mode
        self.force_type: int = 2
        self.force_mode_gain: float = 0.99
        self.force_mode_damp: float = 0.075
        self.force_mode_lim: Tuple[float, ...] = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
        # constants
        self.home_joint_config: Tuple[float, ...] = tuple(self.rtde.r.getActualQ())
        # Load configurations from parameter server
        ConfigServer().load(__name__, self)

    def exit(self) -> None:
        self.rtde.exit()

    ####################################
    #       CONTROLLER FUNCTIONS       #
    ####################################
    def move_home(self) -> None:
        LOGGER.debug("Try to move robot in home configuration")
        success = self.rtde.c.moveJ(
            q=self.home_joint_config,
            speed=self.joint_vel,
            acceleration=self.joint_acc
        )
        if not success:
            LOGGER.warning("Malfunction during movement to the home configuration!")

    def move_l(self, tcp_pose: List[float]) -> None:
        LOGGER.debug(f"Try to move robot to TCP pose {tcp_pose}")
        success = self.rtde.c.moveL(
            pose=tcp_pose,
            speed=self.lin_vel,
            acceleration=self.lin_acc
        )
        if not success:
            cur_pose = self.rtde.r.getActualTCPPose()
            tgt_msg = f"\nTarget pose: {tcp_pose}"
            cur_msg = f"\nCurrent pose: {cur_pose}"
            LOGGER.warning(f"Malfunction during movement to new pose.{tgt_msg}{cur_msg}")

    def move_j(self, q: List[float], vel: float = -1.0, acc: float = -1.0) -> None:
        LOGGER.debug(f"Try to move the robot to new joint configuration {q}")
        speed = self.joint_vel if vel <= 0.0 else vel
        acceleration = self.joint_acc if acc <= 0.0 else acc
        success = self.rtde.c.moveJ(q, speed, acceleration)
        if not success:
            cur_q = self.rtde.r.getActualQ()
            tgt_msg = f"\nTarget joint positions: {q}"
            cur_msg = f"\nCurrent joint positions: {cur_q}"
            LOGGER.warning(f"Malfunction during movement to new joint positions.{tgt_msg}{cur_msg}")

    def servo_l(self, tcp_pose: List[float]) -> None:
        LOGGER.debug(f"Try to move robot to TCP pose {tcp_pose}")
        self.rtde.c.initPeriod()
        success = self.rtde.c.servoL(
            tcp_pose,
            self.lin_servo_vel,
            self.lin_servo_acc,
            self.rtde.dt,
            self.lin_servo_lkh_time,
            self.lin_servo_gain
        )
        self.rtde.c.waitPeriod(self.rtde.dt)
        # Since there is no real time kernel at the moment use python time library
        time.sleep(self.rtde.dt)
        if not success:
            cur_pose = self.rtde.r.getActualTCPPose()
            tgt_msg = f"\nTarget pose: {tcp_pose}"
            cur_msg = f"\nCurrent pose: {cur_pose}"
            LOGGER.warning(f"Malfunction during movement to new pose.{tgt_msg}{cur_msg}")

    def stop_servoing(self) -> None:
        self.rtde.c.servoStop()

    def set_up_force_mode(self, gain: Optional[float] = None, damping: Optional[float] = None) -> None:
        gain_scaling = gain if gain else self.force_mode_gain
        damping_fact = damping if damping else self.force_mode_damp
        self.rtde.c.zeroFtSensor()
        self.rtde.c.forceModeSetGainScaling(gain_scaling)
        self.rtde.c.forceModeSetDamping(damping_fact)

    def force_mode(self,
                   task_frame: List[float],
                   selection_vector: List[int],
                   wrench: List[float],
                   f_mode_type: Optional[int] = None,
                   limits: Optional[Tuple[float, ...]] = None
                   ) -> None:
        """ Function to use the force mode of the ur_rtde API """
        if f_mode_type is None:
            f_mode_type = self.force_type
        if limits is None:
            limits = self.force_mode_lim
        self.rtde.c.forceMode(
            task_frame,
            selection_vector,
            wrench,
            f_mode_type,
            limits
        )
        time.sleep(self.rtde.dt)

    def stop_force_mode(self) -> None:
        """ Function to set robot back in normal position control mode. """
        self.rtde.c.forceModeStop()

    def teach_mode(self) -> None:
        """ Function to enable the free drive mode. """
        self.rtde.c.teachMode()

    def stop_teach_mode(self) -> None:
        """ Function to set robot back in normal position control mode. """
        self.rtde.c.endTeachMode()

    def set_tcp(self, tcp_offset: List[float]) -> None:
        """ Function to set the tcp relative to the tool flange. """
        self.rtde.c.setTcp(tcp_offset)

    ##################################
    #       RECEIVER FUNCTIONS       #
    ##################################
    def get_joint_pos(self) -> List[float]:
        joint_pos: List[float] = self.rtde.r.getActualQ()
        return joint_pos

    def get_joint_vel(self) -> List[float]:
        joint_vel: List[float] = self.rtde.r.getActualQd()
        return joint_vel

    def get_tcp_offset(self) -> List[float]:
        tcp_offset: List[float] = self.rtde.c.getTCPOffset()
        return tcp_offset

    def get_tcp_pose(self) -> List[float]:
        tcp_pose: List[float] = self.rtde.r.getActualTCPPose()
        return tcp_pose

    def get_tcp_force(self) -> List[float]:
        tcp_force: List[float] = self.rtde.r.getActualTCPForce()
        return tcp_force
