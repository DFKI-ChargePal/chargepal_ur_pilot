from __future__ import annotations

# libs
from pathlib import Path
from contextlib import contextmanager

from ur_pilot.ur_pilot import Pilot


# typing
from typing import Iterator


@contextmanager
def connect(config_dir: Path | None = None) -> Iterator[Pilot]:
    pilot = Pilot(config_dir)
    try:
        yield pilot
    finally:
        pilot.disconnect()
