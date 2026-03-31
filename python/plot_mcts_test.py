import json
import os
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

outcomes = {
    "P1Win": 1,
    "P2Win": -1,
    "Draw": 0,
}


def get_outcome(file_path: str) -> int:
    with open(file_path, "r") as f:
        game_log = json.load(f)
        raw_outcome = game_log["outcome"]
    return outcomes[raw_outcome]


def read_files(dir: str) -> Tuple[np.float64, np.float64]:
    files = os.listdir(dir)
    outcome_list = []
    for file_name in files:
        if file_name.endswith((".log", "Store")):
            continue
        file_path = os.path.join(dir, file_name)
        outcome_list.append(get_outcome(file_path))
    return (
        np.mean(outcome_list, dtype=np.float64),
        1.96 * (np.std(outcome_list, dtype=np.float64) / np.sqrt(len(outcome_list))),
    )


def plot_figure(
    fig, xs: list[int], means: np.ndarray, cerrs: np.ndarray, label: str
) -> None:
    ax = fig.gca()
    ax.plot(xs, means, lw=1, label=label)
    ax.fill_between(xs, means - cerrs, means + cerrs, alpha=0.5)
    ax.set_ylim(-1, 1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["16", "", "", "", "256", "512", "1024", "2048", "4096"])
    ax.set_xlabel(r"No. of samples")
    ax.set_ylabel(r"Expected outcome (P1)")


if __name__ == "__main__":
    dirs = [
        "/Users/zaheen/Documents/node_arena/mcts_test_p1",
        "/Users/zaheen/Documents/node_arena/mcts_test_p2",
    ]
    subdirs = [str(2**x) for x in range(4, 13)]
    label = ["Player 1", "Player 2"]

    plt.rcParams.update(
        {"pgf.texsystem": "lualatex", "text.usetex": True, "font.family": "serif"}
    )

    # plt.style.use("ggplot")
    fig = plt.figure(figsize=(6, 4))

    for ind, dir in enumerate(dirs):
        means = np.array([])
        cerrs = np.array([])
        for subdir in subdirs:
            mean, cerr = read_files(os.path.join(dir, subdir))
            means = np.append(means, mean)
            cerrs = np.append(cerrs, cerr)

        xs = [2**x for x in range(4, 13)]
        plot_figure(fig, xs, means, cerrs, label[ind])

    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()
