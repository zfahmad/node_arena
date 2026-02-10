from typing import Any, Generic, Protocol

import numpy as np
import numpy.random as rnd

from python.game_protocols import (
    ActionType,
    GameProtocol,
    Outcomes,
    Player,
    StateProtocol,
)


class Edge(Generic[ActionType]):
    # Edge objects to represent actions taken at a state
    __slots__ = ("action", "N", "W", "outcomes")

    def __init__(self, action: ActionType) -> None:
        self.action: ActionType = action
        self.N: int = 0
        self.W: float = 0.0
        self.outcomes: list["Node"] = []

    @property
    def Q_bar(self) -> float:
        # Q_bar is the sampled average Q-value for selecting the action at this
        # edge from the source node's state
        return self.W / self.N if self.N > 0 else np.inf

    def __repr__(self):
        return f"(Action: {self.action} N: {self.N} Q: {self.Q_bar})"


class Node(Generic[ActionType]):
    # Node objects to represent states in the search tree
    __slots__ = ("state", "V", "N", "edges", "unexpanded_actions")

    def __init__(self, state: StateProtocol) -> None:
        self.state: StateProtocol = state
        self.V: float = 0.0
        self.N: int = 0
        self.edges: list["Edge"] = []
        self.unexpanded_actions: list[Any] = []

    def __repr__(self):
        return f"[State: {self.state.to_string()} V: {self.V} N: {self.N} Actions: {[edge.action for edge in self.edges]}]"


def get_utility(game: GameProtocol, state: StateProtocol) -> float:
    # Returns the outcome of a state as a utility in {-1, 0, 1}
    # If the outcome of a state is a win for the current player at that state,
    # return -1. It is important to keep in mind that the value of a state is
    # the expected value following the policy from the state and not reaching
    # the state.
    outcome: Outcomes = game.get_outcome(state)
    player: Player = state.get_player()
    # print(state.to_string())
    # print(player)
    # print(outcome)
    if outcome == game.Outcomes.P1Win:
        if player == state.Player.One:
            return 1.0
        else:
            return -1.0
    elif outcome == game.Outcomes.P2Win:
        if player == state.Player.Two:
            return 1.0
        else:
            return -1.0
    else:
        return 0.0


class EdgePolicy(Protocol):
    def __call__(self, edges: list[Edge]) -> Edge: ...


class EvaluationFunction(Protocol):
    def __call__(self, game: GameProtocol, state: StateProtocol) -> float: ...


class UCB1(EdgePolicy):
    def __init__(self, seed: int | None = None, C: float = 1.0) -> None:
        self.C: float = C
        self.seed: int | None = seed
        self.rand_ = rnd.default_rng(seed)

    def __call__(self, edges: list[Edge]) -> Edge:
        assert edges, "Cannot run UCB on empty list of edges."

        # Create arrays of sampled averages of Q-values and sample counts
        Q_bars = []
        Ns = []
        for edge in edges:
            Q_bars.append(edge.Q_bar)
            Ns.append(edge.N)
        Q_bars = np.asarray(Q_bars)
        Ns = np.asarray(Ns)
        # print(f"{Q_bars=}")

        # Compute UCB values and choose the edge with the highest value.
        # Ties are randomly broken

        # total_N cannot be less than 1 or else we get divide by zero error in
        # UCB value computation. Setting to 1 should be fine even if no samples
        # are taken since default values of edges should be infinity.
        total_N = max(1, np.sum(Ns))
        Ns = np.clip(Ns, a_min=1, a_max=None)
        # print(f"{total_N=}")
        ucb_values = Q_bars + self.C * np.sqrt(np.log(total_N) / Ns)
        # print(f"{ucb_values=}")
        max_ucb = np.max(ucb_values)
        indices = np.flatnonzero(ucb_values == max_ucb)
        index = self.rand_.choice(indices)

        return edges[index]

    def __repr__(self):
        return f"UCB1(seed:{self.seed},C:{self.C})"


class LCB(EdgePolicy):
    def __init__(self, seed: int | None = None) -> None:
        self.seed: int | None = seed
        self.rand_ = rnd.default_rng(seed)

    def __call__(self, edges: list[Edge]) -> Edge:
        assert edges, "Cannot run LCB on empty list of edges."

        # Create arrays of sampled averages of Q-values and sample counts
        Q_bars = []
        Ns = []
        for edge in edges:
            Q_bars.append(edge.Q_bar)
            Ns.append(edge.N)
        Q_bars = np.asarray(Q_bars)
        Ns = np.asarray(Ns)

        # Compute LCB values and choose the edge with the highest value.
        # Ties are randomly broken
        total_N = np.sum(Ns)
        lcb_values = Q_bars - np.sqrt(np.log(total_N) / Ns)
        # print(lcb_values)
        max_lcb = np.max(lcb_values)
        indices = np.flatnonzero(lcb_values == max_lcb)
        index = self.rand_.choice(indices)

        return edges[index]

    def __repr__(self):
        return f"LCB(seed:{self.seed})"


class EpsilonGreedy(EdgePolicy):
    def __init__(self, seed: int | None = None, epsilon: float = 0.1) -> None:
        self.seed: int | None = seed
        self.rand_ = rnd.default_rng(seed)
        self.epsilon: float = epsilon

    def __call__(self, edges: list[Edge]) -> Edge:
        assert (
            edges
        ), "epsilon_greedy: Cannot select an action from an empty list of edges."

        Q_bars = []
        for edge in edges:
            Q_bars.append(edge.Q_bar)
        Q_bars = np.asarray(Q_bars)

        if self.rand_.uniform(0, 1) < self.epsilon:
            max_q = np.max(Q_bars)
            indices = np.flatnonzero(Q_bars == max_q)
        else:
            indices = list(range(len(edges)))

        index = self.rand_.choice(indices)

        return edges[index]

    def __repr__(self):
        return f"EpsilonGreedy(seed:{self.seed},epsilon:{self.epsilon})"


class MostSampled(EdgePolicy):
    def __init__(self, seed: int | None) -> None:
        self.seed: int | None = seed
        self.rand_ = rnd.default_rng(seed)

    def __call__(self, edges: list[Edge]) -> Edge:
        assert (
            edges
        ), "greatest_counts: Cannot select an action from an empty list of edges."
        Ns = []

        for edge in edges:
            Ns.append(edge.N)
        Ns = np.asarray(Ns)

        max_count = np.max(Ns)
        indices = np.flatnonzero(Ns == max_count)
        index = self.rand_.choice(indices)

        return edges[index]

    def __repr__(self):
        return f"MostSampled(seed:{self.seed})"


class Greedy(EdgePolicy):
    def __init__(self, seed: int | None) -> None:
        self.seed: int | None = seed
        self.rand_ = rnd.default_rng(seed)

    def __call__(self, edges: list[Edge]) -> Edge:
        assert (
            edges
        ), "greatest_counts: Cannot select an action from an empty list of edges."
        Q_bars = []

        for edge in edges:
            Q_bars.append(edge.Q_bar)
        Q_bars = np.asarray(Q_bars)

        max_vals = np.max(Q_bars)
        indices = np.flatnonzero(Q_bars == max_vals)
        index = self.rand_.choice(indices)

        return edges[index]

    def __repr__(self):
        return f"Greedy(seed:{self.seed})"


class RandomRollout(EvaluationFunction):
    def __init__(self, seed: int | None = None, max_depth: int = 50) -> None:
        self.seed: int | None = seed
        self.rand_ = rnd.default_rng(seed)
        self.max_depth: int = max_depth

    def rollout(self, game: GameProtocol, state: StateProtocol, depth: int) -> float:
        if not depth < self.max_depth:
            return 0.0

        if game.is_terminal(state):
            utility: float = get_utility(game, state)
            return -utility

        actions = game.get_actions(state)
        random_action = self.rand_.choice(actions)
        next_state: StateProtocol = game.get_next_state(state, random_action)
        utility: float = -self.rollout(game, next_state, depth + 1)
        return utility

    def __call__(self, game: GameProtocol, state: StateProtocol) -> float:
        depth = 0
        return self.rollout(game, state, depth)

    def __repr__(self):
        return f"RandomRollout(seed:{self.seed},max_depth:{self.max_depth})"
