from typing import Protocol, TypeVar

from python.game_protocols import Game, State

ActionType = TypeVar("ActionType", covariant=True)


class Player(Protocol[ActionType]):
    def select_action(self, game: Game, state: State) -> ActionType: ...
