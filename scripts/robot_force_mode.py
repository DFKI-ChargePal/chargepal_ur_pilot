""" Script to test force mode. """
import logging
import ur_pilot
import argparse
import numpy as np
import camera_kit as ck

from argparse import Namespace

# typing
from typing import Generator, Sequence

LOGGER = logging.getLogger(__name__)


def main(opt: Namespace) -> None:

    # Use a display for user interaction
    # display = ck.Display('Monitor')

    # Connect to pilot
    with ur_pilot.connect() as pilot:
        with pilot.position_control():
            pilot.move_home()
        
        with pilot.force_control():
            pilot.twist_tcp_force_mode('Y', 3.0, 6.0)
            # pilot.twist_tcp_force_mode('Y', 0.0, 20.0)
            
    # Clean up
    # display.destroy()


if __name__ == '__main__':
    """ Script to teach-in a sequence of robot states (joint positions) """
    parser = argparse.ArgumentParser(description="Record a sequence of robot states")
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        ur_pilot.logger.set_logging_level(logging.DEBUG)
    else:
        ur_pilot.logger.set_logging_level(logging.INFO)
    main(args)
