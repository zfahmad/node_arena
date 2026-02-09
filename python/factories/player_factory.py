import importlib
from typing import Mapping

from python.players.player_protocols import PlayerProtocol

FACTORY_REGISTRY = {
    "mcts": (
        "python.factories.player_factories.mcts_player_factory",
        "MCTSPlayerFactory",
    ),
    "random": (
        "python.factories.player_factories.random_player_factory",
        "RandomPlayerFactory",
    ),
    "human": (
        "python.factories.player_factories.human_player_factory",
        "HumanPlayerFactory",
    ),
}


class PlayerFactory:
    def __call__(self, type_: str, raw_config: Mapping) -> PlayerProtocol:
        player_factory_module = importlib.import_module(FACTORY_REGISTRY[type_][0])
        player_factory = getattr(player_factory_module, FACTORY_REGISTRY[type_][1])
        player = player_factory()
        return player(raw_config)


if __name__ == "__main__":
    from pprint import PrettyPrinter
    import yaml
    printer = PrettyPrinter(indent=2, width=40)
    mcts_player_dict = {
        "type_": "mcts",
        "params": {
            "seed": 0,
            "num_samples": 128,
            "gamma": 0.98,
            "tree_policy": {"type_": "ucb1", "seed": 1, "C": 1.5},
            "final_policy": {"type_": "lcb", "seed": 2},
            "eval_func": {"type_": "random_rollout", "seed": 3},
        },
    }

    random_player_dict = {
            "type_": "random",
            "params": {
                "seed": 4
                }
            }

    printer.pprint(mcts_player_dict)
    printer.pprint(random_player_dict)

    PF = PlayerFactory()
    mcts_player = PF(mcts_player_dict["type_"], mcts_player_dict["params"])
    print(mcts_player)
    random_player = PF(random_player_dict["type_"], random_player_dict["params"])
    print(random_player)

    with open("python/test_config.yaml", "r") as f:
        data = yaml.safe_load(f)
    printer.pprint(data)

    mcts_player = PF(data["player_one"]["type_"], data["player_one"]["params"])
    print(mcts_player)
    random_player = PF(data["player_two"]["type_"], data["player_two"]["params"])
    print(random_player)
