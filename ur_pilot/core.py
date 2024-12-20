from __future__ import annotations

# libs
from pathlib import Path
from contextlib import contextmanager

from ur_pilot.ur_pilot import Pilot

# typing
from typing import Generator


@contextmanager
def connect(config_dir: Path | None = None) -> Generator[Pilot, None, None]:
    pilot = Pilot(config_dir)
    try:
        pilot.connect()
        yield pilot
    finally:
        pilot.disconnect()
