# global
import os
import sys
import json
import copy
import logging

# local
from set_up_pilot import _FILE_NAME_CFG_DUMP
from src.utils import query_yes_no

# typing
from typing import Any, Dict


LOGGER = logging.getLogger(__name__)


class ConfigServer:
    
    def __init__(self) -> None:
        # Check if there is a configuration file
        if os.path.isfile(_FILE_NAME_CFG_DUMP):
            # Load configurations
            self.cfg = self._load_cfg_from_file()
        else:
            self.cfg = {}
            LOGGER.warning(f"No configuration found under {_FILE_NAME_CFG_DUMP}!")
            continue_prog = query_yes_no("Continue with default values?", default="no")
            if not continue_prog:
                sys.exit()

    @staticmethod
    def _load_cfg_from_file() -> Dict[str, Any]:
        with open(_FILE_NAME_CFG_DUMP, 'r') as fp:
            cfg: Dict[str, Any] = json.load(fp)
        return cfg

    def load(self, name: str, obj: object) -> None:
        cfg_name = name.split(".")[-1]
        sub_cfg: Dict[str, Any] = {} if cfg_name not in self.cfg else copy.deepcopy(self.cfg[cfg_name])
        for att_name, att_value in sub_cfg.items():
            setattr(obj, att_name, att_value)
