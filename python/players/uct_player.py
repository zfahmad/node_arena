from functools import partial
from typing import Callable, Generic, Type

import numpy as np
import numpy.random as rnd

import python.players.mcts_helpers as mcts
from python.game_protocols import ActionType, Game, State

# from python.players.mcts_helpers import Edge, Node, edge_policy
from python.players.mcts_helpers import *
from python.players.player_protocols import Player


class UCT(Player):
    def __init__(self, tree_policy: EdgePolicy) -> None:
        self.tree_policy = tree_policy

    def select_action(self, game: Game, state: State): ...

    def select(self, edges):
        return self.tree_policy(edges)


if __name__ == "__main__":
    import random

    POLICY_REGISTRY: dict[str, Type[EdgePolicy]] = {
        "ucb": UCB,
        "lcb": LCB,
    }

    def make_policy(name: str, **params) -> EdgePolicy:
        try:
            policy = POLICY_REGISTRY[name]
        except KeyError:
            raise ValueError(f"Non-existent policy: {name}")

        return policy(**params)

    random.seed(0)
    edges = []
    rand = rnd.default_rng(0)
    for i in range(5):
        edge = Edge(i)
        edge.N = random.randint(10, 100)
        edge.W = random.random() * edge.N
        edges.append(edge)
        print(edge)

    # func_name: str = "ucb"
    # tree_policy = make_policy("ucb", C=1.0, seed=0)
    tree_policy = make_policy("lcb", seed=0)
    print(hasattr(tree_policy, "rand"))
    # tree_policy = LCB(seed=0)
    print(type(tree_policy))
    uct = UCT(tree_policy)

    print(uct.tree_policy(edges))
