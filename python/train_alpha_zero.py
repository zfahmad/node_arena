"""
train_alpha_zero.py
Author: Zaheen Ahmad

Plays games in parallel. The number of games played concurrently depend on the
parameter, num_procs, in the config file.

Usage:
    play.py [options] <config-file>

Options:
    -h --help                   # Show this screen
    -o FILE --output=FILE       # Specify output path for game data
    -v --verbose                # Print to stdout
    --resume                    # Restart self-play training [default: False]
    --max-turns=N               # Maximum number of turns to play before ending game [default: 50]
"""

import importlib
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
import uuid
from dataclasses import replace
from multiprocessing import Process, Queue
from multiprocessing.synchronize import Event
from typing import Dict, List

import numpy.random as rand
import yaml

from python.alpha_zero_data_generator import DataGenerator
from python.alpha_zero_learner import Learner
from python.alpha_zero_replay_buffer import ReplayBuffer
from python.configs import *
from python.configure_logging import configure_logging
from python.factories.game_factory import GameFactory
from python.factories.player_factory import PlayerFactory
from python.players.player_protocols import PlayerProtocol
from python.players.puct_inference_server import InferenceClient
from python.players.puct_player import PUCTPlayer

# TODO: Implement code to handle players other than PUCT


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
):
    configure_logging(f"{cfg.output}/game_logs/game_{game_proc_id}.log")
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
    player = create_player(game_proc_id, PF, cfg.player, inference_endpoints)

    # Keep playing games to generate data.
    while True:
        ts = int(time.time() * 1e6)
        fname = f"game_{ts}_{uuid.uuid4().hex}"
        P = DataGenerator(cfg.max_turns, player)
        logging.info(f"[proc {game_proc_id}] Beginning game.")
        P.play(game, state, f"{cfg.output}/self_play/{fname}.tmp")
        logging.info(f"[proc {game_proc_id}] Game end.")
        os.rename(
            f"{cfg.output}/self_play/{fname}.tmp", f"{cfg.output}/self_play/{fname}.h5"
        )
    player.shutdown()  # Currently does nothing --- used if playing finite number of games


def run_inference(
    request_q: Queue,
    response_qs: List[Queue],
    params: InferenceServerConfig,
    log_file: str,
    update_model: Event,
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
        ckpt_path=params.ckpt_dir,
        dims=params.dims,
    )

    inference_server(
        request_q, response_qs, update_model
    )  # runs until all clients shutdown


def run_learner(params: LearnerConfig):
    rb = ReplayBuffer(
        seed=params.seed,
        data_dir=params.working_dir,
        batch_size=params.batch_size,
        buffer_size=params.buffer_size,
    )
    rb.start_indexing_thread()

    # Delay learning until some games are generated.
    buffer_time = 5
    time.sleep(buffer_time)

    learner = Learner(
        params.game_cfg,
        params.model_cfg,
        params.optimizer_cfg,
        params.working_dir,
        params.ckpt_path,
        params.save_interval,
        params.update_model,
    )
    while True:
        batch = rb.get_next_batch()
        learner(batch)


def main():
    from docopt import docopt

    arguments = docopt(__doc__)

    # Generate default output name if not given
    t = time.localtime()
    t_str = "_".join(map(str, [t.tm_year, t.tm_mon, t.tm_hour, t.tm_min, t.tm_sec]))
    output = arguments["--output"]
    log_file = os.path.join(output, t_str)
    configure_logging(f"{log_file}.log")

    if not bool(arguments["--resume"]):
        shutil.rmtree(os.path.join(output, "game_logs"))
        shutil.rmtree(os.path.join(output, "self_play"))
        shutil.rmtree(os.path.join(output, "checkpoints"))

    os.makedirs(os.path.join(output, "game_logs"), exist_ok=True)
    os.makedirs(os.path.join(output, "self_play"), exist_ok=True)
    os.makedirs(os.path.join(output, "checkpoints"), exist_ok=True)

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
        player=PlayerConfig(**raw_cfg["player"]),
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
    update_model = mp.Event()

    # Setup factories
    logging.info("Setting up player and game factories.")
    game_factory = GameFactory()
    player_factory = PlayerFactory()

    # Setup inference servers
    inference_endpoints: Dict[str, InferenceEndpoints] = {}
    inf_processes = []

    logging.info("Setting up inference server process.")
    for server_cfg in cfg.inference_servers:
        request_queue = Queue()
        response_queues = [Queue() for _ in range(cfg.num_procs)]
        inf_process = Process(
            target=run_inference,
            args=(request_queue, response_queues, server_cfg, cfg.output, update_model),
        )
        inf_processes.append(inf_process)
        inference_endpoints[server_cfg.name] = InferenceEndpoints(
            request_queue=request_queue, response_queues=response_queues
        )

    logging.info("Starting inference server...")
    for p in inf_processes:
        p.start()

    # Generate seeds for each actor process
    seeds = generate_random_seeds(cfg.player.params["seed"], cfg.num_procs + 1)

    # Generate per-process configs
    actor_configs = [
        replace(
            cfg,
            player=replace(cfg.player, params={**cfg.player.params, "seed": seeds[i]}),
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

    learner_cfg = LearnerConfig(
        seed=seeds[-1],
        working_dir=cfg.output,
        batch_size=cfg.inference_servers[0].batch_size,
        buffer_size=raw_cfg["learner"]["buffer_size"],
        game_cfg=cfg.game,
        ckpt_path=cfg.inference_servers[0].ckpt_dir,
        save_interval=raw_cfg["learner"]["save_interval"],
        update_model=update_model,
        model_cfg=ModelConfig(
            cfg.game.type_,
            cfg.inference_servers[0].model_type,
            seed=seeds[-1],
            hypers=cfg.game.size,
        ),
        optimizer_cfg=OptimizerConfig(
            raw_cfg["learner"]["optimizer"]["name"],
            raw_cfg["learner"]["optimizer"]["kwargs"],
        ),
    )

    learner_process = Process(target=run_learner, args=(learner_cfg,))

    # Start actors and wait until they finish
    for p in actor_processes:
        p.start()

    learner_process.start()

    for p in actor_processes:
        p.join()
    # Close inference process
    for p in inf_processes:
        p.join()
    learner_process.join()


if __name__ == "__main__":
    main()
