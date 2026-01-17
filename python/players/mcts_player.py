from typing import Callable, Generic, Type

import numpy as np
import numpy.random as rnd

import python.players.mcts_helpers as mcts
from python.game_protocols import ActionType, Game, State

from python.players.mcts_helpers import *
from python.players.player_protocols import Player


class MCTS(Player[ActionType]):
    def __init__(self, tree_policy: EdgePolicy) -> None:
        self.tree_policy = tree_policy

    def select_action(self, game: Game, state: State): ...

    def select(self, edges):
        return self.tree_policy(edges)

    def expand_node(self, game: Game, state: State) -> Node:
        # Creates a new Node for the tree for the given state
        # Generates list of actions available at state
        expanded_node: Node = Node(state)
        actions: list[ActionType] = game.get_actions(state)
        expanded_node.unexpanded_actions = actions
        return expanded_node


if __name__ == "__main__":
    import random

    import python.wrappers.connect_four_wrapper as ttt

    game = ttt.Game()
    state = ttt.State()
    # state.string_to_state("0000200010")
    game.reset(state)
    state.print_board()
    for outcome in game.Outcomes:
        print(outcome)

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
    rand = rnd.default_rng(0)

    # func_name: str = "ucb"
    tree_policy = make_policy("ucb", C=1.0, seed=0)
    evaluation_function = RandomRollout(seed=10, max_depth=30)
    # tree_policy = make_policy("lcb", seed=0)
    uct = MCTS(tree_policy)
    node = uct.expand_node(game, state)
    print(node)
    print(node.unexpanded_actions)
    print(node.edges)
    print(evaluation_function(game, state))
