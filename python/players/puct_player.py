import importlib
import os
from multiprocessing import Process, Queue
from typing import Type

import numpy as np
import numpy.random as rnd
from scipy.special import softmax

from python.game_protocols import ActionType, GameProtocol, StateProtocol
from python.players.mcts_player import Edge, MCTSPlayer, Node
from python.players.puct_inference_server import InferenceClient, InferenceServer


class PUCB:
    def __init__(self, seed: int | None = None, C: float = 1.0) -> None:
        self.C: float = C
        self.seed: int | None = seed
        self.rand_ = rnd.default_rng(seed)

    def __call__(self, edges: list[Edge]) -> Edge:
        assert edges, "Cannot run PUCB on empty list of edges."

        # Create arrays of sampled averages of Q-values and sample counts
        Q_bars = []
        Ns = []
        priors = []
        for edge in edges:
            Ns.append(edge.N)
            if edge.N == 0:
                Q_bars.append(0.0)
            else:
                Q_bars.append(edge.Q_bar)
            priors.append(edge.prior)
        Q_bars = np.asarray(Q_bars)
        Ns = np.asarray(Ns)
        priors = np.asarray(priors)

        # Compute PUCB values and choose the edge with the highest value.
        # Ties are randomly broken

        # total_N cannot be less than 1 or else we get divide by zero error in
        # PUCB value computation. Setting to 1 should be fine even if no samples
        # are taken since default values of edges should be infinity.
        total_N = max(1, np.sum(Ns))
        # Ns = np.clip(Ns, a_min=1, a_max=None)
        ucb_values = Q_bars + self.C * priors * (np.sqrt(total_N) / (1 + Ns))
        max_ucb = np.max(ucb_values)
        indices = np.flatnonzero(ucb_values == max_ucb)
        index = self.rand_.choice(indices)

        return edges[index]

    def __repr__(self):
        return f"PUCB(seed:{self.seed},C:{self.C})"


class SoftArgmax:
    def __init__(self, seed: int | None = None) -> None:
        self.seed: int | None = seed
        self.rand_ = rnd.default_rng(seed)

    def __call__(self, edges: list[Edge], temp: float) -> Edge:
        assert edges

        counts = np.array([edge.N for edge in edges], dtype=float)

        if temp == 0:
            max_count = np.max(counts)
            indices = np.flatnonzero(counts == max_count)
            index = self.rand_.choice(indices)
            return edges[index]

        scaled = counts ** (1.0 / temp)
        total = np.sum(scaled)
        if total == 0:
            probs = np.ones_like(scaled) / len(scaled)
        else:
            probs = scaled / total
        index = self.rand_.choice(len(edges), p=probs)
        return edges[index]

    def __repr__(self):
        return f"SoftArgmax(seed:{self.seed})"


class PUCTPlayer(MCTSPlayer[ActionType]):
    def __init__(
        self,
        inf_client: InferenceClient,
        seed: int | None = None,
        C: float = 1.0,
        num_samples: int = 256,
        gamma: float = 1.0,
        exploitation_threshold: int = 0,
        training: bool = False,
        dirichlet_epsilon: float = 0.1,
        dirichlet_alpha: float = 0.3,
        tree_policy: PUCB | None = None,
        final_policy: SoftArgmax | None = None,
    ) -> None:
        self.inf_client = inf_client
        self.training = training
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        if tree_policy is not None:
            self.tree_policy = tree_policy
        else:
            self.tree_policy = PUCB(seed, C)
        if final_policy is not None:
            self.final_policy = final_policy
        else:
            self.final_policy = SoftArgmax(seed)
        self.exploitation_threshold = exploitation_threshold
        super().__init__(seed, num_samples, gamma, None, None, None)

    def evaluate_node(self, game: GameProtocol, node: Node) -> float:
        value, policy = self.inf_client(node.state)
        mask = np.asarray(game.legal_moves_mask(node.state), dtype=bool)
        if not mask.any():
            node.priors = np.zeros_like(policy)
            return value
        masked_policy = np.where(mask, policy, -np.inf)
        node.priors = softmax(masked_policy)
        return value

    def expand_node(self, game: GameProtocol, node: Node) -> None:
        actions = game.get_actions(node.state)
        for action in actions:
            edge = Edge(action)
            edge.prior = float(node.priors[action])
            node.edges.append(edge)

    def inject_dirichlet_noise(self, node: Node) -> None:
        num_actions = len(node.edges)
        noise = self.rand_.dirichlet([self.dirichlet_alpha] * num_actions)

        for edge, n in zip(node.edges, noise):
            edge.prior = (
                1 - self.dirichlet_epsilon
            ) * edge.prior + self.dirichlet_epsilon * n

        total = sum(edge.prior for edge in node.edges)
        for edge in node.edges:
            edge.prior /= total

    def __call__(
        self,
        game: GameProtocol,
        state: StateProtocol,
        turn: int = 0,
        verbose: bool = False,
    ) -> ActionType | None:
        assert game.get_actions(state), "No actions at state"
        print(turn)

        root = Node(state)
        utility = self.evaluate_node(game, root)
        self.expand_node(game, root)
        root.N = 1
        root.V = utility

        if self.training:
            self.inject_dirichlet_noise(root)

        for _ in range(self.num_samples - 1):
            self.traverse(game, root)

        if verbose:
            self.print_tree(root)
        temp = 1.0
        if turn >= self.exploitation_threshold:
            temp = 0
        action: ActionType = self.final_policy(root.edges, temp).action

        return action

    def __repr__(self):
        return (
            f"PUCT|seed:{self.seed},samples:{self.num_samples},"
            f"gamma:{self.gamma},TP:{self.tree_policy},"
            f"FP:{self.final_policy}"
        )


def run_inference(request_q: Queue, response_qs: list[Queue], params: dict) -> None:
    # TODO: Maybe take model specs as dict?

    # Load model for specified game
    gm = importlib.import_module(f"python.models.{params["game_str"]}_nn")
    model = getattr(gm, params["model_type"])
    m = model(*params["hypers"])

    # Create inference server
    print(f"Creating inference server: {os.getpid()}")
    inference_server = InferenceServer(
        params["batch_size"],
        params["num_actors"],
        gm.create_batch_input,
        gm.create_padding,
        m,
        params["model_ckpt_path"],
        [3, 3],
    )

    # Await requests and process mini-batches until all actors terminate
    # while not shutdown_event.is_set():
    inference_server(request_q, response_qs)


def run_actor(id: int, request_q: Queue, response_q: Queue):
    import random

    import python.wrappers.tic_tac_toe_wrapper as gm

    game = gm.Game()
    state = gm.State()
    game.reset(state)
    state = game.get_next_state(state, random.choice(game.get_actions(state)))
    state.print_board()
    print(f"Creating inference client: {os.getpid()}")
    tree_policy = PUCB(C=1.0, seed=0)
    final_policy = SoftArgmax(seed=0)
    # evaluation_function = RandomRollout(seed=10, max_depth=30)
    client = InferenceClient(id, request_q, response_q)
    puct_agent = PUCTPlayer(
        inf_client=client,
        seed=0,
        C=1.0,
        num_samples=10,
        gamma=1.0,
        exploitation_threshold=10,
        training=True,
        dirichlet_epsilon=0.1,
        dirichlet_alpha=0.3,
        tree_policy=tree_policy,
        final_policy=final_policy,
    )
    print(puct_agent)
    # root = Node(state)
    action = puct_agent(game, state)
    print(action)
    # print(game.get_actions(state))
    # puct_agent.evaluate_node(game, root)
    # puct_agent.expand_node(game, root)
    # for edge in root.edges:
    #     print(edge)
    client.shutdown()


def main():

    import python.wrappers.tic_tac_toe_wrapper as gm

    # from python.players.puct_inference_server import InferenceServer, run_inference

    game = gm.Game()
    print(game.get_id())
    state = gm.State()
    game.reset(state)
    num_actors = 1
    batch_size = 1
    request_q = Queue()
    response_qs = [Queue() for _ in range(num_actors)]
    model_type = "CNN"
    # ckpt_path = "/Users/zaheen/projects/node_arena/python/checkpoints/"
    ckpt_path = "/Users/zaheen/Documents/node_arena/test/checkpoints/60/"
    game_str = "tic_tac_toe"
    model_params = {
        "game_str": game_str,
        "batch_size": batch_size,
        "num_actors": num_actors,
        "model_type": model_type,
        "model_ckpt_path": ckpt_path,
        "hypers": [0, [3, 3]],
    }

    print("Starting inference server.")
    inference_proc = Process(
        target=run_inference,
        args=(request_q, response_qs, model_params),
    )
    inference_proc.start()

    clients = [
        Process(target=run_actor, args=(i, request_q, response_qs[i]))
        for i in range(num_actors)
    ]

    for i in range(num_actors):
        clients[i].start()

    for i in range(num_actors):
        clients[i].join()

    inference_proc.join()


if __name__ == "__main__":
    main()
