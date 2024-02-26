from __future__ import annotations

# libs
import numpy as np
from ur_pilot.ur_robot import URRobot

# typing
from typing import Any
from numpy import typing as npt


class JointState:

    available_states = {
        'na': np.zeros(6, dtype=np.float32),
        'home': np.ones(6, dtype=np.float32),
    }

    def __init__(self, robot: URRobot):
        self.state = 'na'
        self.value = np.zeros(6, np.float32)
        self.robot = robot
        self.check_state()

    def check_state(self) -> None:
        if self.state not in self.available_states.keys():
            raise ValueError(f"Trying to bring arm in an unknown state: {self.state}. "
                             f"Known states are: {self.available_states}")
        if self.state != 'na':
            # TODO: Check against real arm values
            print(f"Check if arm is close to state values for state {self.state}!")

    def check_existence(self, name: str) -> bool:
        exs = False
        # Only work with lowercase strings
        name = name.lower()
        if name in self.available_states.keys():
            exs = True
        return exs

    def add(self, name: str, value: npt.ArrayLike) -> None:
        # Only work with lowercase strings
        name = name.lower()
        if name in self.available_states.keys():
            raise RuntimeError(f"State with name '{name}' already exists.")
        val = np.array(np.reshape(value, 6), dtype=np.float32)
        self.available_states[name] = val

    def get_state(self) -> tuple[str, npt.NDArray[np.float32]]:
        self.check_state()
        return self.state, self.value

    def set_state(self, name: str) -> None:
        # Only work with lowercase strings
        name = name.lower()
        if name not in self.available_states.keys():
            raise RuntimeError(f"Can't find a state with name '{name}'")
        else:
            self.state = name
            self.value = self.available_states[name]
        self.check_state()


class Maneuvers:

    def __init__(self, robot: URRobot, joint_state: JointState):
        self.maneuvers: dict[str, dict[str, Any]] = {}
        self.joint_state = joint_state
        self.robot = robot

    def add(self, name: str, start: str, stop: str, waypoints: list[npt.ArrayLike] | None = None) -> None:
        # Only work with lowercase strings
        name = name.lower()
        if name in self.maneuvers.keys():
            raise RuntimeError(f"Maneuver with name '{name}' already exist.")
        if not self.joint_state.check_existence(start):
            raise ValueError(f"Start state with name '{start}' doesn't exist!")
        if not self.joint_state.check_existence(stop):
            raise ValueError(f"Stop state with name '{stop}' doesn't exist!")
        self.maneuvers[name] = {
            'start': start,
            'stop': stop,
            'waypoints': waypoints
        }

    def execute(self, name: str) -> None:
        # Only work with lowercase strings
        name = name.lower()
        maneuver = self.maneuvers[name]
        if self.joint_state != maneuver['start']:
            raise RuntimeError(f"Arm not in start state. Can't execute maneuver.")
        if maneuver['waypoints'] is None:
            pass
        else:
            pass
