import h5py
from python.game_protocols import GameProtocol, StateProtocol
from python.players.player_protocols import PlayerProtocol
import numpy as np

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
            for edge in root.edges:
                policy[edge.action] = edge.N
            temp = 1.0
            if current_turn >= self.player.exploitation_threshold:  # type: ignore
                temp = 0.0
            action = self.player.final_policy(root.edges, temp).action  # type: ignore
            if temp == 1.0:
                policy = policy ** (1 / temp)
            policy = policy / policy.sum()
            policies.append(policy)

            turn = {
                "turn": current_turn,
                "state": state.to_string(),
                "action": action,
            }
            turns.append(turn)

            state = game.get_next_state(state, action)
            current_turn += 1

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
            state_arrs.append(s.to_array())

        state_arrs = np.array(state_arrs)
        masks = np.array(masks)
        policies = np.array(policies)
        values = np.array(values)

        game_data_dict["outcome"] = game.get_outcome(state).name
        game_data_dict["turns"] = turns
        self.write_data(output_path, state_arrs, masks, policies, values)
