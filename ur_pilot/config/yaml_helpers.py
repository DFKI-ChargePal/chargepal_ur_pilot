from __future__ import annotations

# global
import yaml
from pathlib import Path

# typing
from typing import Any


def load_yaml(file_path: Path) -> dict[str, Any]:
    with file_path.open("r") as filestream:
        try:
            yaml_dict: dict[str, Any] = yaml.safe_load(filestream)
        except Exception as e:
            raise RuntimeError(f"Error while reading {file_path.name} configuration. {e}")
    return yaml_dict


def dump_yaml(data: dict[str, Any], file_path: Path) -> None:
    with file_path.open('wt') as fs:
        yaml.safe_dump(data, fs)
