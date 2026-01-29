import sys
from python.game_protocols import ActionType, GameProtocol, StateProtocol
from python.players.player_protocols import PlayerProtocol


class HumanPlayer(PlayerProtocol[ActionType]):
    def __call__(self, game: GameProtocol, state: StateProtocol) -> ActionType:
        actions: list[ActionType] = game.get_actions(state)
        action_prompt = ""
        for ind, action in enumerate(actions):
            action_prompt += f"\033[95m{ind}: \033[0m({action}) | "
        print(action_prompt)
        action_input: int | str = input("Input action index ('q' to quit): ")
        if action_input == "q":
            sys.exit()
        else:
            action_ind: int = int(action_input)
        return actions[action_ind]

    def __repr__(self):
        return "HumanPlayer"
