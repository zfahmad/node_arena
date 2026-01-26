"""
play.py
Author: Zaheen Ahmad

Defines the class, Play, which is responsible for playing games between two
players.

When called on its own will play out a game between two players specified as
arguments.

List of games currently available:
    tic_tac_toe
    connect_four

List of players currently available:
    random
    mcts|seed,tree_policy,final_policy,default_policy

Usage:
    play.py [options]

Options:
    -h --help                   # Show this screen
    -o FILE --output=FILE       # Specify output path for game data [default: "./"]
    -v --verbose                # Print to stdout
    --max-turns=N               # Maximum number of turns to play before ending game
"""

import json
import os
import time
from typing import Callable

import numpy as np
import numpy.random as rand
import yaml
from factory import GameFactory, PlayerFactory

from python.game_protocols import GameProtocol, Player, StateProtocol
from python.players.player_protocols import PlayerProtocol


class Play:
    def __init__(self, max_turns: int, p1: PlayerProtocol, p2: PlayerProtocol):
        self.max_turns: int = max_turns
        self.players: list[PlayerProtocol] = [p1, p2]

    def play(
        self,
        game: GameProtocol,
        state: StateProtocol,
        output_path: str = "",
        verbose: bool = False,
    ) -> None:
        game_data_dict: dict = {
            "game": game.get_id(),
            "p1": str(self.players[0]),
            "p2": str(self.players[1]),
        }
        turns: list[dict] = []
        current_turn: int = 0
        current_player: Callable[[int], int] = lambda x: x % 2
        while (not game.is_terminal(state)) and (current_turn < self.max_turns):
            player = self.players[current_player(current_turn)]
            action = player(game, state)

            turn = {
                "turn": current_turn,
                "player": current_player(current_turn),
                "state": state.state_to_string(),
                "action": action,
            }
            turns.append(turn)
            if verbose:
                print(
                    f"turn: {current_turn} "
                    "player: {current_player(current_turn)} action: {action}"
                )
                state.print_board()
            state = game.get_next_state(state, action)
            current_turn += 1

        turn = {
            "turn": current_turn,
            "player": current_player(current_turn),
            "state": state.state_to_string(),
            "action": "-",
        }
        turns.append(turn)
        if verbose:
            print(
                f"turn: {current_turn} "
                "player: {current_player(current_turn)} action: "
            )
            state.print_board()
            print(game.get_outcome(state))

        game_data_dict["outcome"] = game.get_outcome(state).name
        game_data_dict["turns"] = turns
        if output_path:
            with open(output_path + ".json", "w", encoding="utf-8") as f:
                json.dump(game_data_dict, f, indent=4)


def main():
    from docopt import docopt

    arguments = docopt(__doc__)
    print(arguments)

    # from python.players.mcts_helpers import LCB, UCB1, RandomRollout
    # from python.players.mcts_player import MCTSPlayer
    # from python.players.randm_player import RandomPlayer
    GF = GameFactory()
    PF = PlayerFactory()
    # print(time.asctime())
    # t = time.localtime()
    # t = "_".join([str(x) for x in [t.tm_year, t.tm_mon, t.tm_hour, t.tm_min, t.tm_sec]])
    game_module = GF("tic_tac_toe")
    game = game_module.Game()
    state = game_module.State()
    # print(game.get_id())
    # PF(arguments['<player-two>'])
    # state = game_module.State()
    # seed = None
    # ucb = UCB1(seed=seed, C=1.0)
    # lcb = LCB(seed=seed)
    # rollout = RandomRollout(seed=seed, max_depth=50)
    # player_1 = MCTSPlayer(
    #     seed=seed,
    #     num_samples=5120,
    #     gamma=0.8,
    #     tree_policy=ucb,
    #     final_policy=lcb,
    #     eval_func=rollout,
    # )
    # player_2 = RandomPlayer(seed=seed)
    # P = Play(100, player_1, player_2)
    # P.play(game, state, t, verbose=True)
    with open("python/test_config.yaml", "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    print(cfg)


if __name__ == "__main__":
    main()
