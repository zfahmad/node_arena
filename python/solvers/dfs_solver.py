from typing import Generic

import h5py
import numpy as np

from python.game_protocols import ActionType, GameProtocol, StateProtocol
from python.players.mcts_helpers import get_utility


class Node(Generic[ActionType]):
    # Node objects to represent states in the search tree
    __slots__ = ("state", "V")

    def __init__(self, state: StateProtocol) -> None:
        self.state: StateProtocol = state
        self.V: float = -np.inf

    def __repr__(self) -> str:
        return f"[State: {self.state.state_to_string()} V: {self.V}]"


class DepthFirstSearch:
    def __init__(self) -> None:
        self.cache: dict[str, float] = {}

    def traverse(self, game: GameProtocol, node: Node) -> float:
        if game.is_terminal(node.state):
            node.V = get_utility(game, node.state)
            self.cache[node.state.state_to_string()] = node.V
            return -node.V

        utility = self.cache.get(node.state.state_to_string())
        if utility:
            return -utility

        actions = game.get_actions(node.state)
        for action in actions:
            child_state = game.get_next_state(node.state, action)
            child_node = Node(child_state)
            utility = self.traverse(game, child_node)
            if utility > node.V:
                node.V = utility

        self.cache[node.state.state_to_string()] = node.V

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
    # state.string_to_state("0053444100000000002c0b0200000000167")
    # state.string_to_state("0102001001")
    # state.print_board()
    # print(state.get_player())
    # arrs = np.array(state.to_array())
    # print(arrs.reshape(2, 6, 7))
    # print(arrs.reshape(2, 3, 3))
    # string_to_input_rep(state.state_to_string())

    dfs = DepthFirstSearch()
    dfs(game, state)

    file_path = "./python/ttt_dataset.h5"

    states = []
    values = []
    for key in dfs.cache.keys():
        state.string_to_state(key)
        states.append(state.to_array())
        values.append(dfs.cache[key])

    states = np.array(states)
    values = np.array(values)
    indices = np.arange(states.shape[0])
    indices = np.random.permutation(indices)

    with h5py.File(file_path, "w") as f:
        training = f.create_group("training")
        testing = f.create_group("testing")
        training.create_dataset("states", data=states[indices[:4478]])
        training.create_dataset("values", data=values[indices[:4478]])
        testing.create_dataset("states", data=states[indices[4478:]])
        testing.create_dataset("values", data=values[indices[4478:]])
