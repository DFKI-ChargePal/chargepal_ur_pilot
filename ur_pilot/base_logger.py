import logging


def set_logging_level(level: int) -> None:
    """ Helper to configure logging

    Args:
        level: Logging level

    Returns:
        None
    """
    logging.basicConfig(format='%(levelname)s: %(message)s', level=level)


def get_logging_level() -> int:
    """ Helper function to get root logging level """
    return logging.root.level
