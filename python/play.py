import json
import os
import time

import numpy as np
import numpy.random as rand

from python.game_protocols import Game, Player, State
from python.players.player_protocols import PlayerProtocol


class Play:
    def __init__(self, max_turns: int, p1: PlayerProtocol, p2: PlayerProtocol):
        self.max_turns: int = max_turns
        self.players: list[PlayerProtocol] = [p1, p2]

    def play(
        self, game: Game, state: State, output_path: str = "", verbose: bool = False
    ) -> None:
        game_data_dict: dict = {
            "game": game.get_id(),
            "p1": str(self.players[0]),
            "p2": str(self.players[1]),
        }
        turns: list[dict] = []
        current_turn: int = 0
        current_player = lambda x: x % 2
        while (not game.is_terminal(state)) and (current_turn < self.max_turns):
            player = self.players[current_player(current_turn)]
            action = player(game, state)
            turn = {
                "turn": current_turn,
                "player": current_player(current_turn),
                "state": state.state_to_string(),
                "action": action,
            }

            if verbose:
                print(
                    f"turn: {current_turn} player: {current_player(current_turn)} action: {action}"
                )
                state.print_board()

            turns.append(turn)
            state = game.get_next_state(state, action)
            current_turn += 1
        turn = {
            "turn": current_turn,
            "player": current_player(current_turn),
            "state": state.state_to_string(),
            "action": "-",
        }
        if verbose:
            print(
                f"turn: {current_turn} player: {current_player(current_turn)} action: "
            )
            state.print_board()
            print(game.get_outcome(state))
        turns.append(turn)

        game_data_dict["outcome"] = game.get_outcome(state).name
        game_data_dict["turns"] = turns
        if output_path:
            with open(output_path + ".json", "w", encoding="utf-8") as f:
                json.dump(game_data_dict, f, indent=4)


if __name__ == "__main__":
    import python.wrappers.connect_four_wrapper as ttt
    from python.players.mcts_helpers import LCB, UCB1, RandomRollout
    from python.players.mcts_player import MCTSPlayer
    from python.players.random_player import RandomPlayer

    print(time.asctime())
    t = time.localtime()
    t = "_".join([str(x) for x in [t.tm_year, t.tm_mon, t.tm_hour, t.tm_min, t.tm_sec]])
    game = ttt.Game()
    state = ttt.State()
    seed = 42
    ucb = UCB1(seed=seed, C=1.0)
    lcb = LCB(seed=seed)
    rollout = RandomRollout(seed=seed, max_depth=30)
    player_1 = MCTSPlayer(
        seed=seed,
        num_samples=5_120,
        tree_policy=ucb,
        final_policy=lcb,
        eval_func=rollout,
    )
    player_2 = RandomPlayer(seed=seed)
    P = Play(100, player_1, player_2)
    P.play(game, state, t, verbose=True)
