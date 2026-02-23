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
from dataclasses import dataclass, field, replace
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional

import numpy.random as rand
import yaml

from python.configure_logging import configure_logging
from python.factories.game_factory import GameFactory
from python.factories.player_factory import PlayerFactory
from python.play import Play
from python.players.player_protocols import PlayerProtocol


@dataclass
class PlayerConfig:
    type_: str
    params: dict
    model: Optional[str] = None


@dataclass
class GameConfig:
    type_: str
    size: list[int]
    initial_state: str = ""


@dataclass
class InferenceServerConfig:
    game_str: str
    name: str
    type_: str
    num_actors: int
    batch_size: int
    model_type: str
    ckpt_dir: str
    seed: int
    dims: list[int]
    model_hypers: List[Any]


@dataclass
class Config:
    output: str
    verbose: bool
    max_turns: int
    num_procs: int
    game: GameConfig
    player_one: PlayerConfig
    player_two: PlayerConfig
    inference_servers: List[InferenceServerConfig] = field(default_factory=list)


@dataclass
class InferenceEndpoints:
    request_queue: Queue
    response_queues: List[Queue]


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

    client = None
    if player_cfg.model:
        endpoints = inference_endpoints[player_cfg.model]
        client = InferenceClient(
            proc_id,
            endpoints.request_queue,
            endpoints.response_queues[proc_id],  # only this player's queue
        )

    player = PF(player_cfg.type_, player_cfg.params)
    logging.info(f"[proc {proc_id}] Created player: {player}")
    return player


def run_game(
    game_proc_id: int,
    GF: GameFactory,
    PF: PlayerFactory,
    cfg: Config,
    inference_endpoints: Dict[str, InferenceEndpoints],
):
    configure_logging(f"{cfg.output}_{game_proc_id}.log")
    logging.info(f"[proc {game_proc_id}] Loading game.")

    game_module = GF(cfg.game.type_)
    game = game_module.Game()
    logging.info(f"[proc {game_proc_id}] Loaded game: {game.get_id()}")
    print(game.get_id())

    # Initialize state
    if cfg.game.size:
        state = game_module.State(cfg.game.size[0], cfg.game.size[1])
    else:
        state = game_module.State()

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
    logging.info(f"[proc {game_proc_id}] Beginning game.")
    P.play(game, state, f"{cfg.output}_{game_proc_id}", cfg.verbose)
    logging.info(f"[proc {game_proc_id}] Game end.")


def run_inference(
    request_q: Queue,
    response_qs: List[Queue],
    params: InferenceServerConfig,
    log_file: str,
):
    configure_logging(f"{log_file}.log")
    logging.info(f"[server {params.name}] Creating inference model.")

    game_module = importlib.import_module(f"python.models.{params.game_str}_nn")
    Model = getattr(game_module, params.model_type)
    model = Model(params.seed, params.dims)

    logging.info(f"[server {params.name}] Creating inference server")
    sm = importlib.import_module(f"python.players.{params.type_}_inference_server")
    inference_server = sm.InferenceServer(
        batch_size=params.batch_size,
        num_actors=params.num_actors,
        create_batch_input=game_module.create_batch_input,
        create_padding=game_module.create_padding,
        model=model,
        ckpt_dir=params.ckpt_dir,
        dims=params.dims
    )

    inference_server(request_q, response_qs)  # runs until all clients shutdown


def main():
    from docopt import docopt

    arguments = docopt(__doc__)

    # Generate default output name if not given
    t = time.localtime()
    t_str = "_".join(map(str, [t.tm_year, t.tm_mon, t.tm_hour, t.tm_min, t.tm_sec]))
    output = arguments["--output"] or t_str
    configure_logging(f"{output}.log")

    # Load config file
    logging.info(f"Loading config file: {arguments['<config-file>']}")
    try:
        with open(arguments["<config-file>"]) as f:
            raw_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file does not exist: {arguments['<config-file>']}")
        sys.exit(1)

    # Build dataclass config
    cfg = Config(
        output=output,
        verbose=arguments.get("--verbose", False),
        max_turns=int(arguments.get("--max-turns", raw_cfg.get("max_turns", 100))),
        num_procs=raw_cfg["num_procs"],
        game=GameConfig(**raw_cfg["game"]),
        player_one=PlayerConfig(**raw_cfg["player_one"]),
        player_two=PlayerConfig(**raw_cfg["player_two"]),
        inference_servers=[
            InferenceServerConfig(
                **s,
                game_str=raw_cfg["game"]["type_"],
                dims=raw_cfg["game"]["size"],
                num_actors=raw_cfg["num_procs"],
            )
            for s in raw_cfg.get("inference_servers", [])
        ],
    )

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
                i,
                game_factory,
                player_factory,
                actor_configs[i],
                inference_endpoints,
            ),
        )
        for i in range(cfg.num_procs)
    ]

    for p in actor_processes:
        p.start()
    for p in actor_processes:
        p.join()
    for p in inf_processes:
        p.join()


if __name__ == "__main__":
    main()
