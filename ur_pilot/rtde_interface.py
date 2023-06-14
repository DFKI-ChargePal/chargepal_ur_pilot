""" This file defines the ur_rtde interface class. """
# global
import logging
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

# local
from ur_pilot.config_server import ConfigServer


LOGGER = logging.getLogger(__name__)


class RTDEInterface:

    def __init__(self, ip_address: str, robot_freq: float, verbose: bool = False) -> None:
        # Default configurations
        self.ur_ip = ip_address
        self.ur_freq = robot_freq
        self.verbose = verbose
        # Load configurations from parameter server
        ConfigServer().load(__name__, self)
        self.dt = 1.0/self.ur_freq
        # noinspection PyArgumentList
        self.c = RTDEControl(self.ur_ip, self.ur_freq)
        # noinspection PyArgumentList
        self.r = RTDEReceive(self.ur_ip, self.ur_freq)

    def exit(self) -> None:
        LOGGER.debug("Disconnect to all interfaces.")
        # Stop RTDE control scripts
        self.c.stopScript()
        # Disconnect to the RTDE interfaces
        self.c.disconnect()
        self.r.disconnect()
