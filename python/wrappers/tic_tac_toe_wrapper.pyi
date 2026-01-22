import enum


class Player(enum.Enum):
    One = 0

    Two = 1

class State:
    def __init__(self) -> None: ...

    def print_board(self) -> None: ...

    def state_to_string(self) -> str: ...

    def string_to_state(self, arg: str, /) -> None: ...

    def get_player(self) -> Player: ...

    def get_opponent(self) -> Player: ...

    def set_player(self, arg: Player, /) -> None: ...

    class Player(enum.Enum):
        One = 0

        Two = 1

class Outcomes(enum.Enum):
    NonTerminal = 0

    P1Win = 1

    P2Win = 2

    Draw = 3

class Game:
    def __init__(self) -> None: ...

    def get_id(self) -> str: ...

    def reset(self, arg: State, /) -> None: ...

    def get_actions(self, arg: State, /) -> list[int]: ...

    def apply_action(self, arg0: State, arg1: int, /) -> int: ...

    def get_next_state(self, arg0: State, arg1: int, /) -> State: ...

    def is_winner(self, arg0: State, arg1: Player, /) -> bool: ...

    def is_draw(self, arg: State, /) -> bool: ...

    def is_terminal(self, arg: State, /) -> bool: ...

    def get_outcome(self, arg: State, /) -> Outcomes: ...

    class Outcomes(enum.Enum):
        NonTerminal = 0

        P1Win = 1

        P2Win = 2

        Draw = 3
