"""
parallel_play.py
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

import importlib
import logging
import os
import sys
import time
from multiprocessing import Process, Queue
from typing import Any

import numpy.random as rand

from python.play import Play
from python.configure_logging import configure_logging
from python.factories.game_factory import GameFactory
from python.factories.player_factory import PlayerFactory


def generate_random_seeds(base_seed: int, num_seeds: int):
    seed_sequence = rand.SeedSequence(base_seed)
    child_seeds = seed_sequence.spawn(num_seeds)
    return [int(seed.generate_state(1)[0]) for seed in child_seeds]


def run_game(
    game_proc_id: int,
    GF: GameFactory,
    PF: PlayerFactory,
    config_dict: dict,
    log_file: str,
    **kwargs: Any,
):
    configure_logging(f"{log_file}_{game_proc_id}.log")
    logging.info(f"Loading game.")
    game_module = GF(config_dict["game"]["type_"])
    game = game_module.Game()
    logging.info(f"Loaded game: {game.get_id()}.")
    print(game.get_id())
    # print(f"Creating inference client: {os.getpid()}")
    # client = InferenceClient(
    #     game_proc_id, kwargs["puct_infq"], kwargs["puct_resqs"][game_proc_id]
    # )
    # client(state)
    # client.shutdown()
    logging.info(f"Successfully loaded: {game.get_id()}")
    if config_dict.get("size", None):
        state = game_module.State(
            config_dict["game"]["size"][0], config_dict["game"]["size"][1]
        )
    else:
        state = game_module.State()
    if config_dict["game"]["initial_state"] == "":
        game.reset(state)
        logging.info(f"Created initial state.")
    else:
        state.string_to_state(config_dict["game"]["initial_state"])
        logging.info(f"Created state from specification.")

    config_dict["player_one"]["params"]["seed"] = config_dict["player_one"]["seeds"][
        game_proc_id
    ]
    logging.info(f"Creating first player...")
    player_one = PF(
        config_dict["player_one"]["type_"], config_dict["player_one"]["params"]
    )
    logging.info(f"Created: {player_one}")

    config_dict["player_two"]["params"]["seed"] = config_dict["player_two"]["seeds"][
        game_proc_id
    ]
    logging.info(f"Creating second player...")
    player_two = PF(
        config_dict["player_two"]["type_"], config_dict["player_two"]["params"]
    )
    logging.info(f"Created: {player_two}")

    # Play game
    P = Play(int(config_dict["max_turns"]), player_one, player_two)
    logging.info(f"Beginning game.")
    P.play(
        game, state, config_dict["output"] + f"_{game_proc_id}", config_dict["verbose"]
    )
    logging.info(f"Game end.")


def run_inference(
    request_q: Queue, response_qs: list[Queue], params: dict, log_file: str
) -> None:
    # TODO: Maybe take model specs as dict?

    # Load model for specified game
    configure_logging(f"{log_file}_server.log")
    game_module = importlib.import_module(f"python.models.{params["game_str"]}_nn")
    Model = getattr(game_module, params["model_type"])
    model = Model(params["seed"], *params["hypers"])

    # Create inference server
    print(f"Creating inference server: {os.getpid()}")
    sm = importlib.import_module(f"python.players.{params["type_"]}_inference_server")
    inference_server = sm.InferenceServer(
        params["batch_size"],
        params["num_actors"],
        game_module.create_batch_input,
        game_module.create_padding,
        model,
        params["model_ckpt_path"],
    )

    # Await requests and process mini-batches until all actors terminate
    # while not shutdown_event.is_set():
    inference_server(request_q, response_qs)


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
    # inference_factory = PUCTServerFactory()

    # Load config file
    logging.info("Loading config file...")
    try:
        with open(arguments["<config-file>"], "r") as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file: {arguments['<config-file>']}, does not exist.")
        sys.exit()
    logging.info(f"Successfully loaded: {arguments['<config-file>']}")

    config_dict.update(
        {
            "output": output,
            "verbose": arguments.get("--verbose", False),
            "max_turns": arguments.get("--max-turns"),
        }
    )

    # Create queues and inference servers
    extra_kwargs = {}
    inf_processes = []
    inf_server_specs = config_dict.get("inference_servers")
    for inf_server_spec in inf_server_specs:
        inference_queue = Queue()
        response_queues = [Queue() for _ in range(config_dict["num_procs"])]
        inf_server_config = {
            "type_": inf_server_spec["type_"],
            "game_str": config_dict["game"]["type_"],
            "batch_size": inf_server_spec["batch_size"],
            "num_actors": config_dict["num_procs"],
            "model_type": inf_server_spec["model_type"],
            "model_ckpt_path": inf_server_spec["ckpt_dir"],
            "seed": inf_server_spec["seed"],
            "hypers": inf_server_spec["model_hypers"],
        }
        inf_processes.append(
            Process(
                target=run_inference,
                args=(inference_queue, response_queues, inf_server_config),
                kwargs={"log_file": output},
            )
        )
        extra_kwargs.update(
            {
                f"{inf_server_spec["name"]}_infq": inference_queue,
                f"{inf_server_spec["name"]}_resqs": response_queues,
            }
        )

    for inf_proc in inf_processes:
        inf_proc.start()

    num_procs: int = config_dict["num_procs"]

    # Generate seeds for each process using the seed in the config file as the
    # base seed
    proc_seeds_p1 = generate_random_seeds(
        config_dict["player_one"]["params"]["seed"], num_procs
    )
    proc_seeds_p2 = generate_random_seeds(
        config_dict["player_two"]["params"]["seed"], num_procs
    )
    config_dict["player_one"]["seeds"] = proc_seeds_p1
    config_dict["player_two"]["seeds"] = proc_seeds_p2

    logging.info(f"Creating actor processes.")
    actor_processes = [
        Process(
            target=run_game,
            args=(
                game_proc_id,
                game_factory,
                player_factory,
                config_dict,
            ),
            kwargs={**extra_kwargs, "log_file": output},
        )
        for game_proc_id in range(num_procs)
    ]

    logging.info(f"Starting actor processes.")
    for actor in actor_processes:
        actor.start()

    logging.info(f"Waiting for actor processes to complete.")
    for actor in actor_processes:
        actor.join()

    logging.info(f"Waiting for server processes to complete.")
    for inf_proc in inf_processes:
        inf_proc.join()


if __name__ == "__main__":
    main()
