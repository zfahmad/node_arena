from typing import Type

import numpy as np
import numpy.random as rnd

from python.game_protocols import ActionType, GameProtocol, StateProtocol
from python.players.mcts_helpers import *
from python.players.player_protocols import PlayerProtocol


class MCTSPlayer(PlayerProtocol[ActionType]):
    def __init__(
        self,
        seed: int | None = None,
        num_samples: int = 256,
        gamma: float = 1.0,
        tree_policy: EdgePolicy | None = None,
        final_policy: EdgePolicy | None = None,
        eval_func: EvaluationFunction | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.gamma = gamma
        self.seed = seed
        self.rand_ = rnd.default_rng(seed=seed)

        if tree_policy:
            self.tree_policy = tree_policy
        else:
            self.tree_policy = UCB1(None, C=1.0)

        if final_policy:
            self.final_policy = final_policy
        else:
            self.final_policy = LCB(None)

        if eval_func:
            self.eval_func = eval_func
        else:
            self.eval_func = RandomRollout(None, max_depth=30)

    def select_action(self, node: Node) -> Edge:
        return self.final_policy(node.edges)

    def traverse(self, game: GameProtocol, node: Node) -> float:
        # End traversal if terminal state is reached.
        # Return the outcome of the terminal state.
        # node.state.print_board()
        # print(game.get_outcome(node.state))
        if game.is_terminal(node.state):
            utility: float = get_utility(game, node.state)
            node.N += 1
            return -utility

        # If the node has not been visited before, evaluate the node following
        # the evaluation function.
        if node.N == 0:
            utility: float = self.evaluate_node(game, node)
            self.expand_node(game, node)
            node.N += 1
            node.V += utility
            return -utility

        # The node has been visited before and is not a terminal node

        # Add an unexpanded action to the list of expanded edges:
        # Select a random action, create an edge with the action, then append to
        # list of edges.
        if node.unexpanded_actions:
            random_ind: np.int64 = self.rand_.integers(len(node.unexpanded_actions))
            unexpanded_action: ActionType = node.unexpanded_actions.pop(random_ind)
            node.edges.append(Edge(unexpanded_action))

        # Choose an edge from the list of expanded edges using the tree policy
        selected_edge: Edge = self.select(node.edges)

        if not selected_edge.outcomes:
            new_state: StateProtocol = game.get_next_state(
                node.state, selected_edge.action
            )
            # new_child: Node = self.expand_node(game, new_state)
            new_child: Node = Node(new_state)
            selected_edge.outcomes.append(new_child)

        # Outcomes is a list to accomodate MCTS with double progressive
        # widening. All search algorithmsm will share the same data structures.
        # In this MCTS the list should only contain one outcome per edge as
        # transactions are deterministic.
        child: Node = selected_edge.outcomes[0]

        # Traverse the tree following the selected edge.
        utility: float = self.traverse(game, child)

        # Update path statistics
        selected_edge.W += utility
        selected_edge.N += 1
        node.V += utility
        node.N += 1

        return -self.gamma * utility

    def select(self, edges: list[Edge]) -> Edge:
        return self.tree_policy(edges)

    # def expand_node(self, game: GameProtocol, state: StateProtocol) -> Node:
    #     # Creates a new Node for the tree for the given state
    #     # Generates list of actions available at state
    #     expanded_node: Node = Node(state)
    #     actions: list[ActionType] = game.get_actions(state)
    #     expanded_node.unexpanded_actions = actions
    #     return expanded_node
    def expand_node(self, game: GameProtocol, node: Node) -> None:
        actions: list[ActionType] = game.get_actions(node.state)
        node.unexpanded_actions = actions

    def evaluate_node(self, game: GameProtocol, node: Node) -> float:
        return self.eval_func(game, node.state)

    def print_tree(self, root: Node, depth: int = 0) -> None:
        indent = "    "
        print(f"{depth * indent}{root}")
        if not root.edges:
            return None
        for edge in root.edges:
            print(f"{depth * indent + '  '}{edge}")
            if edge.outcomes:
                self.print_tree(edge.outcomes[0], depth + 1)

    def __call__(
        self, game: GameProtocol, state: StateProtocol, verbose: bool = False
    ) -> ActionType | None:
        assert game.get_actions(state), "No actions at state"

        root = Node(state)
        self.expand_node(game, root)
        for _ in range(self.num_samples):
            self.traverse(game, root)

        if verbose:
            self.print_tree(root)
        action: ActionType = self.select_action(root).action

        return action

    def __repr__(self):
        return (
            f"MCTS|seed:{self.seed},samples:{self.num_samples},"
            f"gamma:{self.gamma},TP:{self.tree_policy},"
            f"FP:{self.final_policy},DP:{self.eval_func}"
        )


if __name__ == "__main__":
    import random

    import python.wrappers.tic_tac_toe_wrapper as ttt

    game = ttt.Game()
    state = ttt.State()
    game.reset(state)
    state.string_to_state("0000200010")
    state.print_board()

    POLICY_REGISTRY: dict[str, Type[EdgePolicy]] = {
        "ucb1": UCB1,
        "lcb": LCB,
    }

    def make_policy(name: str, **params) -> EdgePolicy:
        try:
            policy = POLICY_REGISTRY[name]
        except KeyError:
            raise ValueError(f"Non-existent policy: {name}")

        return policy(**params)

    random.seed(0)
    rand = rnd.default_rng(0)

    # func_name: str = "ucb"
    # state.string_to_state("2000201010")
    # state.print_board()
    tree_policy = make_policy("ucb", C=1.0, seed=0)
    final_policy = make_policy("lcb", seed=0)
    evaluation_function = RandomRollout(seed=10, max_depth=30)
    # tree_policy = make_policy("lcb", seed=0)
    uct = MCTSPlayer(0, 1024, 0.98, tree_policy, final_policy, evaluation_function)
    print(uct)
    # uct(game, state, True)
    # state.string_to_state("2000201111")
    # state.print_board()
    # print(game.get_outcome(state))
    while not game.is_terminal(state):
        action = uct(game, state)
        print(action)
        if action is None:
            raise RuntimeError("MCTS did not select an action.")
        state = game.get_next_state(state, action)
        state.print_board()
    # node = uct.expand_node(game, state)
    # for action in node.unexpanded_actions:
    #     node.edges.append(Edge(action))
    # print(node)
    # for edge in node.edges:
    #     print(f"  \u221f {edge}")
    # # print(node.unexpanded_actions)
    # # print(node.edges)
    # utility = evaluation_function(game, state)
    # print(utility)
    # print(type(state.get_player()))
    # outcome = game.get_outcome(state)
    # print(outcome_to_utility(game, state))
