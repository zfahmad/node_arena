"""
random.py

Author: Zaheen Ahmad

A random player that selects a random action at each game state.
"""

import random

from python.game_protocols import ActionType, GameProtocol, StateProtocol
from python.players.player_protocols import PlayerProtocol


class RandomPlayer(PlayerProtocol[ActionType]):
    def __init__(self, seed: int | None) -> None:
        self.seed = seed
        self._rand = random.Random(seed)

    def __call__(self, game: GameProtocol, state: StateProtocol) -> ActionType:
        actions: list[ActionType] = game.get_actions(state)
        return self._rand.choice(actions)

    def __repr__(self):
        return f"RandomPlayer|seed:{self.seed}"


if __name__ == "__main__":
    import python.wrappers.connect_four_wrapper as c4
    import python.wrappers.tic_tac_toe_wrapper as ttt

    game: GameProtocol = ttt.Game()
    state = ttt.State()
    game.reset(state)
    player = RandomPlayer(seed=1)
    actions = game.get_actions(state)
    state.print_board()
    action = player(game, state)
    state = game.get_next_state(state, action)
    state.print_board()
    player = state.get_player()
    print(game)
