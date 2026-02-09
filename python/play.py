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
    mcts

Usage:
    play.py [options] <config-file>

Options:
    -h --help                   # Show this screen
    -o FILE --output=FILE       # Specify output path for game data
    -v --verbose                # Print to stdout
    --max-turns=N               # Maximum number of turns to play before ending game [default: 50]
"""

import json
import logging
import os
import sys
import time
from typing import Callable

from python.configure_logging import configure_logging
from python.factories.game_factory import GameFactory
from python.factories.player_factory import PlayerFactory
from python.game_protocols import GameProtocol, StateProtocol
from python.players.player_protocols import PlayerProtocol
from python.players.puct_inference_server import InferenceClient


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
        if verbose:
            print(f"turn: {current_turn} " f"player: {current_player(current_turn)}")
            state.print_board()

        # Begin playing loop
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

            state = game.get_next_state(state, action)
            current_turn += 1

            if verbose:
                print(
                    f"turn: {current_turn} "
                    f"player: {current_player(current_turn)} action: {action}"
                )
                state.print_board()

        turn = {
            "turn": current_turn,
            "player": current_player(current_turn),
            "state": state.state_to_string(),
            "action": "-",
        }
        turns.append(turn)

        if verbose:
            print(game.get_outcome(state))

        game_data_dict["outcome"] = game.get_outcome(state).name
        game_data_dict["turns"] = turns
        if output_path:
            with open(output_path + ".json", "w", encoding="utf-8") as f:
                json.dump(game_data_dict, f, indent=4)


def main():
    import yaml
    from docopt import docopt

    arguments = docopt(__doc__)

    # Create name for output file.
    t = time.localtime()
    t = "_".join([str(x) for x in [t.tm_year, t.tm_mon, t.tm_hour, t.tm_min, t.tm_sec]])
    output = arguments["--output"]
    if output == "":
        output = t

    if output:
        configure_logging(output + ".log")

    # Instantiate factories
    game_factory = GameFactory()
    player_factory = PlayerFactory()

    # Load config file
    logging.info("Loading config file...")
    try:
        with open(arguments["<config-file>"], "r") as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file: {arguments['<config-file>']}, does not exist.")
        sys.exit()
    logging.info(f"Successfully loaded: {arguments['<config-file>']}")

    # Load game and create players
    logging.info("Loading game...")
    game_module = game_factory(config_dict["game"]["type_"])
    game = game_module.Game()
    logging.info(f"Successfully loaded: {game.get_id()}")

    # Create state of specified size
    if config_dict["game"].get("size", None):
        state = game_module.State(
            config_dict["game"]["size"][0], config_dict["game"]["size"][1]
        )
    else:
        state = game_module.State()

    # Initialize state if specified, else start a new game
    if config_dict["game"]["initial_state"] == "":
        game.reset(state)
        logging.info("Created initial state.")
    else:
        state.string_to_state(config_dict["game"]["initial_state"])
        logging.info("Created state from specification.")

    # Create players
    logging.info("Creating first player...")
    player_one = player_factory(
        config_dict["player_one"]["type_"], config_dict["player_one"]["params"]
    )
    logging.info(f"Created: {player_one}")
    logging.info("Creating second player...")
    player_two = player_factory(
        config_dict["player_two"]["type_"], config_dict["player_two"]["params"]
    )
    logging.info(f"Created: {player_two}")

    # Play game
    P = Play(int(arguments["--max-turns"]), player_one, player_two)
    logging.info("Beginning game.")
    P.play(game, state, output, verbose=bool(arguments["--verbose"]))
    logging.info("Game end.")


if __name__ == "__main__":
    main()
