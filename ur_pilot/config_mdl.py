""" Configuration class definitions. """
from __future__ import annotations

# global
import tomli
from pathlib import Path

import tomli_w
from pydantic import BaseModel, Field, NoneStr

# typing
from typing import Any, List, Optional


class MissingConfigError(Exception):
    """ Configuration file not found.
    """
    def __init__(self, path: Path):  # noqa: D107
        self.message = f"Configuration file not found under path {path.absolute()}."
        super().__init__(self.message)


class ToolModel(BaseModel):
    """ Data model for the end-effector
    """
    mass: float = 0.0
    com: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    tip_frame: List[float] = Field(default_factory=lambda: 6 * [0.0])
    sense_frame: List[float] = Field(default_factory=lambda: 6 * [0.0])


class ForceMode(BaseModel):
    """ Data model for robot force mode.
    """
    mode: int = 2
    gain: float = 0.99
    damping: float = 0.075
    tcp_speed_limits: List[float] = Field(default_factory=lambda: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])


class MotionMode(BaseModel):
    """ Data model for robot motion mode.
    """
    error_scale = 5000.0
    force_limit = 50.0
    Kp: List[float] = Field(default_factory=lambda: [100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    Kd: List[float] = Field(default_factory=lambda: [0.99, 0.99, 0.99, 0.99, 0.99, 0.99])


class HybridMode(BaseModel):
    """ Data model for robot hybrid mode.
    """
    error_scale = 100.0
    force_limit = 20.0
    Kp_force: List[float] = Field(default_factory=lambda: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    Kd_force: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Kp_motion: List[float] = Field(default_factory=lambda: [100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    Kd_motion: List[float] = Field(default_factory=lambda: [0.99, 0.99, 0.99, 0.99, 0.99, 0.99])


class FTSensor(BaseModel):
    """ Force torque sensor configuration
    """
    adapter: str = 'enx3c18a01939b0'
    slave_pos: int = 0
    filter_sinc_length: int = 512
    filter_fir: bool = True
    filter_fast: bool = False
    filter_chop: bool = False
    ft_bias: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class Pilot(BaseModel):
    """ Base data model for robot configuration.
    """
    tool_model: ToolModel = ToolModel()
    ft_sensor: Optional[FTSensor] = None
    force_mode: ForceMode = ForceMode()
    motion_mode: MotionMode = MotionMode()
    hybrid_mode: HybridMode = HybridMode()


class Config(Pilot):
    """ Data model for robot configuration.
    """
    name: NoneStr = None
    description: NoneStr = None
    author: NoneStr = None
    email: NoneStr = None
    pilot: Pilot


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


def write_toml(config: dict[str, Any], file_path: Path) -> None:
    """ Write robot configuration to TOML file.

    Args:
        config:    Configuration dictionary
        file_path: Path to the new TOML configuration file.

    """
    with file_path.open(mode='wb') as fp:
        tomli_w.dump(config, fp)
