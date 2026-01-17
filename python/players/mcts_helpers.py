import numpy as np
import numpy.random as rnd
from typing import Generic, Protocol

from python.game_protocols import ActionType, State

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
        return self.W / self.N if self.N > 0 else 0.0

    def __repr__(self):
        return f"Action: {self.action} N: {self.N} Q: {self.Q_bar}"


class Node(Generic[ActionType]):
    # Node objects to represent states in the search tree
    __slots__ = ("state", "N", "actions")

    def __init__(self, state: State) -> None:
        self.state: State = state
        self.N: int = 0
        self.actions: list["Edge"] = []

    def __repr__(self):
        return f"State: {self.state.state_to_string} N: {self.N}"


class EdgePolicy(Protocol):
    def __call__(self, edges: list[Edge]) -> Edge: ...



class UCB(EdgePolicy):
    def __init__(self, C: float, seed: int):
        self.C = C
        self.rand = rnd.default_rng(seed)

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

        # Compute UCB values and choose the edge with the highest value.
        # Ties are randomly broken
        total_N = np.sum(Ns)
        ucb_values = Q_bars + self.C * np.sqrt(np.log(Ns) / total_N)
        max_ucb = np.max(ucb_values)
        indices = np.flatnonzero(ucb_values == max_ucb)
        index = self.rand.choice(indices)

        return edges[index]


class LCB(EdgePolicy):
    def __init__(self, seed: int):
        self.rand = rnd.default_rng(seed)

    def __call__(self, edges: list[Edge]):
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
        lcb_values = Q_bars - np.sqrt(np.log(Ns) / total_N)
        max_lcb = np.max(lcb_values)
        indices = np.flatnonzero(lcb_values == max_lcb)
        index = self.rand.choice(indices)

        return edges[index]
