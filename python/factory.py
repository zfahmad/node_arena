"""
factory.py

Author: Zaheen Ahmad

Factory function to handle loading games.
"""

import importlib
import sys
from types import ModuleType


def factory(module_name_str: str) -> ModuleType:
    try:
        game_module = importlib.import_module("wrapper." + module_name_str + "_wrapper")
    except ModuleNotFoundError:
        print("Game not found.", file=sys.stderr)
        exit(1)
    return game_module
