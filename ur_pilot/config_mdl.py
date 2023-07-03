""" Configuration class definitions. """
from __future__ import annotations

# global
import tomli
from pathlib import Path
from pydantic import BaseModel, Field, NoneStr

# typing
from typing import Any, List, Optional


class MissingConfigError(Exception):
    """ Configuration file not found.
    """
    def __init__(self, path: Path):  # noqa: D107
        self.message = f"Configuration file not found under path {path.absolute()}."
        super().__init__(self.message)


class Joints(BaseModel):
    """ Data model for robot joint values.
    """
    vel: float = 0.2
    max_vel: float = 1.0
    acc: float = 0.5
    max_acc: float = 1.0


class TCP(BaseModel):
    """ Data model for robot TCP values.
    """
    vel: float = 0.2
    max_vel: float = 1.0
    acc: float = 0.5
    max_acc: float = 1.0


class Tool(BaseModel):
    """ Data model for the end-effector
    """
    mass: float = 0.0
    com: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class Servo(BaseModel):
    """ Data model for robot servo values. 
    """
    vel: float = 0.5
    acc: float = 0.5
    gain: float = 200.0
    lkh_time: float = 0.2


class ForceMode(BaseModel):
    """ Data model for robot force mode.
    """
    mode: int = 2
    gain: float = 0.99
    damping: float = 0.075
    tcp_speed_limits: List[float] = Field(default_factory=lambda: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  


class FTSensor(BaseModel):
    """ Base data model for the force torque sensor configuration
    """
    adapter: str = 'enx3c18a01939b0'
    slave_pos: int = 0
    filter_sinc_length: int = 512
    filter_fir: bool = True
    filter_fast: bool = False
    filter_chop: bool = False
    ft_bias: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_tcp2ft: List[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    p_tcp2ft: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.035])


class Robot(BaseModel):
    """ Base data model for robot configuration.
    """
    ip_address: str = "192.168.13.42"
    rtde_freq: float = 500.0
    home_radians: Optional[List[float]] = None
    joints: Joints = Joints()
    tcp: TCP = TCP()
    tool: Tool = Tool()
    servo: Servo = Servo()
    ft_sensor: Optional[FTSensor] = FTSensor()
    force_mode: ForceMode = ForceMode()


class Config(Robot):
    """ Data model for robot configuration.
    """
    name: NoneStr = None
    description: NoneStr = None
    author: NoneStr = None
    email: NoneStr = None
    robot: Robot


def read_toml(config_file: Path) -> dict[str, Any]:
    """
    Read robot configuration from TOML file.

    Args:
        config_file: Path to TOML configuration file.

    Raises:
        MissingConfigError: Error raised if configuration file is not found.

    Returns:
        Robot configuration dictionary.
    """
    if config_file.exists():
        with open(config_file, "rb") as f:
            config_dict = tomli.load(f)
            return config_dict
    else:
        raise MissingConfigError(config_file)
