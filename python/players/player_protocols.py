from typing import Protocol, TypeVar

from python.game_protocols import Game, State

ActionType = TypeVar("ActionType", covariant=True)


class PlayerProtocol(Protocol[ActionType]):
    def __call__(self, game: Game, state: State) -> ActionType | None: ...
