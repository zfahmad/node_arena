import os
import numpy as np
import numpy.random as rand

from python.game_protocols import Game, Player, State
from python.players.player_protocols import PlayerProtocol


class Play:
    def __init__(self, max_turns: int, p1: PlayerProtocol, p2: PlayerProtocol):
        self.max_turns: int = max_turns
        self.players: list[PlayerProtocol] = [p1, p2]

    def play(self, game: Game, state: State, output_path: str=""):
        current_turn: int = 0
        current_player = lambda x: x % 2
        state.print_board()
        while (not game.is_terminal(state)) and (current_turn < self.max_turns):
            player = self.players[current_player(current_turn)]
            print(f"Turn: {current_turn} Player: {player}")
            # actions = game.get_actions(state)
            # print(actions)
            action = player(game, state)
            state = game.get_next_state(state, action)
            print(state.state_to_string())
            current_turn += 1


if __name__ == "__main__":
    import python.wrappers.connect_four_wrapper as ttt
    from python.players.mcts_helpers import LCB, UCB1, RandomRollout
    from python.players.mcts_player import MCTSPlayer
    from python.players.random_player import RandomPlayer

    game = ttt.Game()
    state = ttt.State()
    seed = 42
    ucb = UCB1(seed=seed, C=1.0)
    lcb = LCB(seed=seed)
    rollout = RandomRollout(seed=seed, max_depth=30)
    player_1 = MCTSPlayer(
        seed=seed, num_samples=5_120, tree_policy=ucb, final_policy=lcb, eval_func=rollout
    )
    player_2 = RandomPlayer(seed=seed)
    P = Play(100, player_1, player_2)
    P.play(game, state)
