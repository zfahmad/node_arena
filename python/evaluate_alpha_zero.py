"""
evaluate_alpha_zero.py
Author: Zaheen Ahmad

Evaluates AlphaZero against a reference agent (MCTS). Evaluates each checkpoint
saved when training AlphaZero against a specified agent.

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
from dataclasses import dataclass, field, replace
from multiprocessing import Process, Queue
from typing import Dict, List

import numpy.random as rand
import yaml

from python.configs import (
    GameConfig,
    InferenceEndpoints,
    InferenceServerConfig,
    ModelConfig,
    PlayerConfig,
)
from python.configure_logging import configure_logging
from python.factories.game_factory import GameFactory
from python.factories.player_factory import PlayerFactory
from python.play import Play
from python.players.player_protocols import PlayerProtocol
from python.players.puct_inference_server import InferenceClient
from python.players.puct_player import PUCTPlayer
from python.plot_alpha_zero import plot_results


@dataclass
class Config:
    output: str
    verbose: bool
    max_turns: int
    num_procs: int
    games_per_proc: int
    game: GameConfig
    player_one: PlayerConfig
    player_two: PlayerConfig
    inference_servers: list[InferenceServerConfig] = field(default_factory=list)


def generate_random_seeds(base_seed: int, num_seeds: int) -> List[int]:
    seed_sequence = rand.SeedSequence(base_seed)
    child_seeds = seed_sequence.spawn(num_seeds)
    return [int(seed.generate_state(1)[0]) for seed in child_seeds]


def create_player(
    proc_id: int,
    PF: PlayerFactory,
    player_cfg: PlayerConfig,
    inference_endpoints: Dict[str, InferenceEndpoints],
) -> PlayerProtocol:
    logging.info(f"[proc {proc_id}] Creating player {player_cfg.type_}...")

    player = PF(player_cfg.type_, player_cfg.params)
    client = None
    if player_cfg.server:
        endpoints = inference_endpoints[player_cfg.server]
        client = InferenceClient(
            proc_id,
            endpoints.request_queue,
            endpoints.response_queues[proc_id],  # only this player's queue
        )
        # INFO: I may want to rethink this. Maybe have a protocol for inference
        # based players if I want to use other types of learning agents.
        if isinstance(player, PUCTPlayer):
            player.inf_client = client

    logging.info(f"[proc {proc_id}] Created player: {player}")
    return player


def run_game(
    game_proc_id: int,
    GF: GameFactory,
    PF: PlayerFactory,
    cfg: Config,
    inference_endpoints: Dict[str, InferenceEndpoints],
) -> None:
    configure_logging(f"{cfg.output}_{game_proc_id}.log")
    logging.info(f"[proc {game_proc_id}] Loading game.")

    game_module = GF(cfg.game.type_)
    game = game_module.Game(*cfg.game.params)
    logging.info(f"[proc {game_proc_id}] Loaded game: {game.get_id()}")

    # Initialize state
    # Use a specified starting state if provided.
    # Else use initial game state.
    state = game_module.State(*cfg.game.size)
    if cfg.game.initial_state == "":
        game.reset(state)
        logging.info(f"[proc {game_proc_id}] Created initial state.")
    else:
        state.from_string(cfg.game.initial_state)
        logging.info(f"[proc {game_proc_id}] Created state from specification.")

    # Create players
    player_one = create_player(game_proc_id, PF, cfg.player_one, inference_endpoints)
    player_two = create_player(game_proc_id, PF, cfg.player_two, inference_endpoints)

    # Play game
    P = Play(cfg.max_turns, player_one, player_two)
    for ite in range(cfg.games_per_proc):
        logging.info(f"[proc {game_proc_id}] Beginning game.")
        P.play(
            game, state, os.path.join(cfg.output, f"{game_proc_id}_{ite}"), cfg.verbose
        )
        logging.info(f"[proc {game_proc_id}] Game end.")
    for player in [player_one, player_two]:
        player.shutdown()


def run_inference(
    request_q: Queue,
    response_qs: List[Queue],
    params: InferenceServerConfig,
    log_file: str,
) -> None:
    configure_logging(f"{log_file}.log")
    logging.info(f"[server {params.name}] Creating inference model.")

    game_module = importlib.import_module(f"python.models.{params.game_str}_nn")
    Model = getattr(game_module, params.model_cfg.name)
    model = Model(params.model_cfg.seed, **params.model_cfg.hypers)

    logging.info(f"[server {params.name}] Creating inference server")
    sm = importlib.import_module(f"python.players.{params.type_}_inference_server")
    inference_server = sm.InferenceServer(
        batch_size=params.batch_size,
        num_actors=params.num_actors,
        create_batch_input=game_module.create_batch_input,
        create_padding=game_module.create_padding,
        model=model,
        ckpt_path=params.ckpt_dir,
        dims=params.dims,
    )

    inference_server(request_q, response_qs)  # runs until all clients shutdown


def evaluate_agent(cfg: Config):
    # Setup factories
    game_factory = GameFactory()
    player_factory = PlayerFactory()

    # Setup inference servers
    inference_endpoints: Dict[str, InferenceEndpoints] = {}
    inf_processes = []

    for server_cfg in cfg.inference_servers:
        request_queue = Queue()
        response_queues = [Queue() for _ in range(cfg.num_procs)]
        inf_process = Process(
            target=run_inference,
            args=(request_queue, response_queues, server_cfg, cfg.output),
        )
        inf_processes.append(inf_process)
        inference_endpoints[server_cfg.name] = InferenceEndpoints(
            request_queue=request_queue, response_queues=response_queues
        )

    for p in inf_processes:
        p.start()

    # Generate seeds for each actor process
    seeds_p1 = generate_random_seeds(cfg.player_one.params["seed"], cfg.num_procs)
    seeds_p2 = generate_random_seeds(cfg.player_two.params["seed"], cfg.num_procs)

    # Generate per-process configs
    actor_configs = [
        replace(
            cfg,
            player_one=replace(
                cfg.player_one, params={**cfg.player_one.params, "seed": seeds_p1[i]}
            ),
            player_two=replace(
                cfg.player_two, params={**cfg.player_two.params, "seed": seeds_p2[i]}
            ),
        )
        for i in range(cfg.num_procs)
    ]

    # Launch actor processes
    actor_processes = [
        Process(
            target=run_game,
            args=(
                proc_id,
                game_factory,
                player_factory,
                actor_configs[proc_id],
                inference_endpoints,
            ),
        )
        for proc_id in range(cfg.num_procs)
    ]

    for p in actor_processes:
        p.start()
    for p in actor_processes:
        p.join()
    for p in inf_processes:
        p.join()


def main():
    from docopt import docopt

    arguments = docopt(__doc__)

    # Generate default output name if not given
    t = time.localtime()
    t_str = "_".join(map(str, [t.tm_year, t.tm_mon, t.tm_hour, t.tm_min, t.tm_sec]))
    output = arguments["--output"] or t_str
    configure_logging(f"{output}.log")
    os.makedirs(os.path.join(output, "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(output, "evaluation", "player_one"), exist_ok=True)
    os.makedirs(os.path.join(output, "evaluation", "player_two"), exist_ok=True)
    os.makedirs(os.path.join(output, "plots"), exist_ok=True)

    # Load config file
    logging.info(f"Loading config file: {arguments['<config-file>']}")
    try:
        with open(arguments["<config-file>"]) as f:
            raw_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file does not exist: {arguments['<config-file>']}")
        sys.exit(1)

    # Get list of saved checkpoint paths
    ckpt_paths = [
        ckpt_path
        for ckpt_path in os.listdir(os.path.join(output, "checkpoints"))
        if not ckpt_path.endswith((".DS_Store", "latest"))
    ]
    num_base_seeds = len(ckpt_paths) * len(raw_cfg["num_samples"])
    base_seeds_p1 = generate_random_seeds(
        raw_cfg["player"]["params"]["seed"], num_base_seeds
    )
    base_seeds_p2 = generate_random_seeds(
        raw_cfg["mcts"]["params"]["seed"], num_base_seeds
    )
    inf_seeds = generate_random_seeds(raw_cfg["master_seed"], 2)
    seed_idx = 0

    for ckpt_dir in ckpt_paths:
        print()
        inference_servers = [
            InferenceServerConfig(
                game_str=raw_cfg["game"]["type_"],
                dims=raw_cfg["game"]["size"],
                num_actors=raw_cfg["num_procs"],
                seed=inf_seeds[-1],
                name=s["name"],
                type_=s["type_"],
                batch_size=s["batch_size"],
                ckpt_dir=os.path.join(output, "checkpoints", ckpt_dir),
                model_cfg=ModelConfig(
                    type_=raw_cfg["game"]["type_"],
                    name=s["model_cfg"]["name"],
                    seed=inf_seeds[-2],
                    hypers=s["model_cfg"]["hypers"],
                ),
            )
            for s in raw_cfg.get("inference_servers", [])
        ]

        for perspective in ["player_one", "player_two"]:
            for num_samples in raw_cfg["num_samples"]:
                eval_output_dir = os.path.join(
                    output, f"evaluation/{perspective}/{ckpt_dir}", str(num_samples)
                )
                os.makedirs(eval_output_dir, exist_ok=True)
                puct_cfg = PlayerConfig(**raw_cfg["player"])
                puct_cfg.params["seed"] = base_seeds_p1[seed_idx]
                mcts_cfg = PlayerConfig(**raw_cfg["mcts"])
                mcts_cfg.params["seed"] = base_seeds_p2[seed_idx]
                mcts_cfg.params["num_samples"] = num_samples

                player_one_cfg = puct_cfg
                player_two_cfg = mcts_cfg
                if perspective == "player_two":
                    player_one_cfg, player_two_cfg = player_two_cfg, player_one_cfg

                # Build dataclass config
                cfg = Config(
                    output=eval_output_dir,
                    verbose=arguments.get("--verbose", False),
                    max_turns=int(
                        arguments.get("--max-turns", raw_cfg.get("max_turns", 100))
                    ),
                    num_procs=raw_cfg["num_procs"],
                    games_per_proc=raw_cfg["games_per_proc"],
                    game=GameConfig(**raw_cfg["game"]),
                    player_one=player_one_cfg,
                    player_two=player_two_cfg,
                    inference_servers=inference_servers,
                )

                evaluate_agent(cfg)
        seed_idx += 1

    plot_results(
        os.path.join(output, "evaluation/player_one"),
        raw_cfg["num_samples"],
        os.path.join(output, "plots/player_one"),
    )
    plot_results(
        os.path.join(output, "evaluation/player_two"),
        raw_cfg["num_samples"],
        os.path.join(output, "plots/player_two"),
    )


if __name__ == "__main__":
    main()
