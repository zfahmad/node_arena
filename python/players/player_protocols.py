from typing import Protocol, TypeVar

from python.game_protocols import GameProtocol, StateProtocol

ActionType = TypeVar("ActionType", covariant=True)


class PlayerProtocol(Protocol[ActionType]):
    def __call__(self, game: GameProtocol, state: StateProtocol) -> ActionType | None: ...
    def __repr__(self) -> str: ...
