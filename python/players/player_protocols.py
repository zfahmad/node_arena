from typing import Protocol, TypeVar, Optional

from python.game_protocols import GameProtocol, StateProtocol
from python.players.puct_inference_server import InferenceClient

ActionType = TypeVar("ActionType", covariant=True)


class PlayerProtocol(Protocol[ActionType]):
    def shutdown(self) -> None: ...
    def __call__(self, game: GameProtocol, state: StateProtocol) -> ActionType | None: ...
    def __repr__(self) -> str: ...
