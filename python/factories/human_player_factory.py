from python.players.human_player import HumanPlayer


class HumanPlayerFactory:
    def make_human_player(self) -> HumanPlayer:
        return HumanPlayer()

    def __call__(self, raw_config) -> HumanPlayer:
        player = self.make_human_player()
        return player
