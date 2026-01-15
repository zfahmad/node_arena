import numpy as np
import numpy.random as rnd
from typing import Generic

from python.game_protocols import ActionType, Game, State


class UCT(Generic[ActionType]):
    def __init__(self, seed: int | None, C: float=1.0):
        if seed is not None:
            self._rand = rnd.default_rng(seed)
        else:
            self._rand = rnd.default_rng()
        self.C = C

    class Node:
        def __init__(self, state: State):
            self.state: State = state
            self.uct_value: float = np.inf
            self.visit_count: int = 0
