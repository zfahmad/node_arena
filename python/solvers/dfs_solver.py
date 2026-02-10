from dataclasses import dataclass, field
from typing import Generic

import h5py
import numpy as np

from python.game_protocols import ActionType, GameProtocol, StateProtocol
from python.players.mcts_helpers import get_utility


class Node(Generic[ActionType]):
    # Node objects to represent states in the search tree
    __slots__ = ("state", "V", "policy")

    def __init__(self, state: StateProtocol) -> None:
        self.state: StateProtocol = state
        self.V: float = -np.inf
        self.policy: np.ndarray = np.zeros(9)

    def __repr__(self) -> str:
        return f"[State: {self.state.to_string()} V: {self.V} P: {self.policy}]"


@dataclass
class StateLabel:
    value: float
    policy: np.ndarray


class DepthFirstSearch:
    def __init__(self) -> None:
        self.cache: dict[str, StateLabel] = {}

    def traverse(self, game: GameProtocol, node: Node) -> float:
        if game.is_terminal(node.state):
            node.V = get_utility(game, node.state)
            self.cache[node.state.to_string()] = StateLabel(
                value=node.V, policy=np.zeros(9)
            )
            return -node.V

        label = self.cache.get(node.state.to_string(), None)
        if label:
            return -label.value

        actions = game.get_actions(node.state)
        action_values = np.zeros(9)
        for action in actions:
            child_state = game.get_next_state(node.state, action)
            child_node = Node(child_state)
            utility = self.traverse(game, child_node)
            action_values[action] = utility
            if utility > node.V:
                node.V = utility

        policy = (action_values == np.max(action_values)).astype(float)
        policy = np.divide(policy, policy.sum(), where=(policy.sum() != 0.0))
        policy = policy * game.legal_moves_mask(node.state)
        self.cache[node.state.to_string()] = StateLabel(node.V, policy)

        return -node.V

    def __call__(self, game: GameProtocol, state: StateProtocol) -> float:
        root = Node(state)
        return self.traverse(game, root)


if __name__ == "__main__":
    import python.wrappers.tic_tac_toe_wrapper as G

    # import python.wrappers.connect_four_wrapper as G
    game = G.Game()
    state = G.State()
    game.reset(state)
    print(state.to_compact())
    state.from_string("0000000011")
    # state.from_string("0053444100000000002c0b0200000000167")
    state.from_string("0201021011")
    state.print_board()
    print(state.to_compact())
    state.from_compact([0, 0])
    state.print_board()
    state.from_compact([256, 0])
    state.print_board()
    # print(state.to_array())
    # actions = game.get_actions(state)
    # print(actions)
    # print(state.get_player())
    # arrs = np.array(state.to_array())
    # print(arrs.reshape(2, 6, 7))
    # print(arrs.reshape(2, 3, 3))
    # string_to_input_rep(state.to_string())

    # dfs = DepthFirstSearch()
    # dfs(game, state)
    #
    # file_path = "./python/ttt_dataset.h5"
    #
    # states = []
    # values = []
    # policies = []
    # for key in dfs.cache.keys():
    #     state.from_string(key)
    #     # states.append(state.)
    #     values.append(dfs.cache[key].value)
    #     policies.append(dfs.cache[key].policy)

    # states = np.array(states)
    # values = np.array(values)
    # policies = np.array(policies)
    # print(values.shape)
    # print(policies.shape)

    # print(np.reshape(np.sum(states[-15:], axis=1), (-1, 3, 3)))
    # print(states[-15:])
    # print(np.reshape(policies[-15:], (-1, 3, 3)))
    # indices = np.arange(states.shape[0])
    # indices = np.random.permutation(indices)
    #
    # with h5py.File(file_path, "w") as f:
    #     training = f.create_group("training")
    #     testing = f.create_group("testing")
    #     training.create_dataset("states", data=states[indices[:4478]])
    #     training.create_dataset("values", data=values[indices[:4478]])
    #     training.create_dataset("policies", data=policies[indices[:4478]])
    #     testing.create_dataset("states", data=states[indices[4478:]])
    #     testing.create_dataset("values", data=values[indices[4478:]])
    #     testing.create_dataset("policies", data=policies[indices[4478:]])
