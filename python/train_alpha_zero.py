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
    --max-turns=N               # Maximum number of turns to play before ending game [default: 50]
"""

import importlib
import logging
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from multiprocessing import Process, Queue
from typing import Dict, List, Optional

import h5py
import numpy as np
import numpy.random as rand
import yaml

from python.configure_logging import configure_logging
from python.factories.game_factory import GameFactory
from python.factories.player_factory import PlayerFactory
from python.game_protocols import GameProtocol, StateProtocol
from python.players.player_protocols import PlayerProtocol
from python.players.puct_inference_server import InferenceClient
from python.players.puct_player import PUCTPlayer

# TODO: Implement code to handle players other than PUCT


@dataclass
class PlayerConfig:
    type_: str
    params: dict
    server: Optional[str] = None


@dataclass
class GameConfig:
    type_: str
    params: list[int]
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


@dataclass
class Config:
    output: str
    verbose: bool
    max_turns: int
    num_procs: int
    game: GameConfig
    player: PlayerConfig
    inference_servers: List[InferenceServerConfig] = field(default_factory=list)


@dataclass
class InferenceEndpoints:
    request_queue: Queue
    response_queues: List[Queue]


class DataGenerator:
    def __init__(self, max_turns: int, player: PlayerProtocol):
        self.max_turns: int = max_turns
        self.player: PlayerProtocol = player

    def write_data(self, output_path, state_arrs, masks, policies, values):
        f = h5py.File(output_path, "w")
        f.create_dataset("states", data=state_arrs)
        f.create_dataset("masks", data=masks)
        f.create_dataset("policies", data=policies)
        f.create_dataset("values", data=values)
        f.close()

    # NOTE: CURRENTLY ONLY HANDLES PUCT!!!
    def play(
        self,
        game: GameProtocol,
        state: StateProtocol,
        output_path: str = "",
    ) -> None:
        game_data_dict: dict = {
            "game": game.get_id(),
            "player": str(self.player),
        }
        turns: list[dict] = []
        current_turn: int = 0
        states: list[StateProtocol] = []
        masks = []
        policies = []

        # Begin playing loop
        while (not game.is_terminal(state)) and (current_turn < self.max_turns):
            states.append(state)
            root = self.player.run_tree_search(game, state)  # type: ignore

            mask = game.legal_moves_mask(state)
            policy = np.zeros_like(mask)
            masks.append(mask)
            policies.append(policy)
            for edge in root.edges:
                policy[edge.action] = edge.N
            temp = 1.0
            if current_turn >= self.player.exploitation_threshold:  # type: ignore
                temp = 0.0
            action = self.player.final_policy(root.edges, temp).action  # type: ignore

            turn = {
                "turn": current_turn,
                "state": state.to_string(),
                "action": action,
            }
            turns.append(turn)

            state = game.get_next_state(state, action)
            current_turn += 1

        state.print_board()
        turn = {
            "turn": current_turn,
            "state": state.to_string(),
            "action": "-",
        }
        turns.append(turn)

        values = []
        state_arrs = []
        outcome = game.get_outcome(state)

        if outcome == game.Outcomes.P1Win:
            value = 1.0
        elif outcome == game.Outcomes.P2Win:
            value = -1.0
        else:
            value = 0.0

        for s in states:
            if s.get_player() == s.Player.One:
                values.append([value])
            else:
                values.append([-value])
            state_arrs.append(state.to_array())

        state_arrs = np.array(state_arrs)
        masks = np.array(masks)
        policies = np.array(policies)
        values = np.array(values)

        game_data_dict["outcome"] = game.get_outcome(state).name
        game_data_dict["turns"] = turns
        self.write_data(output_path, state_arrs, masks, policies, values)


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

    # Play games
    for _ in range(10):
        ts = int(time.time() * 1e6)
        fname = f"game_{ts}_{uuid.uuid4().hex}"
        P = DataGenerator(cfg.max_turns, player)
        logging.info(f"[proc {game_proc_id}] Beginning game.")
        P.play(game, state, f"{cfg.output}/self_play/{fname}.tmp")
        logging.info(f"[proc {game_proc_id}] Game end.")
        os.rename(
            f"{cfg.output}/self_play/{fname}.tmp", f"{cfg.output}/self_play/{fname}.h5"
        )
    player.shutdown()


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
        ckpt_path=params.ckpt_dir,
        dims=params.dims,
    )

    inference_server(request_q, response_qs)  # runs until all clients shutdown


def main():
    from docopt import docopt

    arguments = docopt(__doc__)

    # Generate default output name if not given
    t = time.localtime()
    t_str = "_".join(map(str, [t.tm_year, t.tm_mon, t.tm_hour, t.tm_min, t.tm_sec]))
    output = arguments["--output"]
    log_file = output + t_str
    configure_logging(f"{log_file}.log")
    os.makedirs(output + "game_logs", exist_ok=True)
    if os.path.exists(output + "self_play"):
        shutil.rmtree(output + "self_play")
    os.makedirs(output + "self_play", exist_ok=True)

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
    seeds = generate_random_seeds(cfg.player.params["seed"], cfg.num_procs)

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

    # Start actors and wait until they finish
    for p in actor_processes:
        p.start()
    for p in actor_processes:
        p.join()

    # Close inference process
    for p in inf_processes:
        p.join()


if __name__ == "__main__":
    main()
