"""
game_factory.py

Author: Zaheen Ahmad

Factory function to handle loading games.
"""

import importlib
import sys
from types import ModuleType


class GameFactory:
    def __call__(self, module_name_str: str) -> ModuleType:
        try:
            game_module = importlib.import_module(
                "wrappers." + module_name_str + "_wrapper"
            )
        except ModuleNotFoundError:
            print("Game not found.", file=sys.stderr)
            exit(1)
        return game_module
