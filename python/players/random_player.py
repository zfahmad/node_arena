"""
random.py

Author: Zaheen Ahmad

A random player that selects a random action at each game state.
"""

import random

from python.game_protocols import ActionType, Game, State
from python.players.player_protocols import Player


class RandomPlayer(Player[ActionType]):
    def __init__(self, seed: int | None) -> None:
        if seed is not None:
            self._rand = random.Random(seed)
        else:
            self._rand = random.Random()

    def select_action(self, game: Game, state: State) -> ActionType:
        actions: list[ActionType] = game.get_actions(state)
        return self._rand.choice(actions)


if __name__ == "__main__":
    import python.wrappers.connect_four_wrapper as c4
    import python.wrappers.tic_tac_toe_wrapper as ttt

    game: Game = ttt.Game()
    state = ttt.State()
    game.reset(state)
    player = RandomPlayer(seed=1)
    actions = game.get_actions(state)
    state.print_board()
    action = player.select_action(game, state)
    state = game.get_next_state(state, action)
    state.print_board()
    player = state.get_player()
    print(game)
