import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import backends

plt.rcParams.update(
    {"pgf.texsystem": "lualatex", "text.usetex": True, "font.family": "serif"}
)

outcomes = {"P1Win": 1, "P2Win": -1, "Draw": 0, "NonTerminal": 0}


def get_outcome(file_path: str) -> int:
    with open(file_path, "r") as f:
        game_log = json.load(f)
        raw_outcome = game_log["outcome"]
    return outcomes[raw_outcome]


def generate_plot(
    iterations: list[int], labels: list[int], data: np.ndarray, output_path: str
):
    for i in range(len(labels)):
        # plt.plot(iterations, data[i, :, 0])
        # plt.fill_between(
        #     iterations,
        #     data[i, :, 0] - data[i, :, 1],
        #     data[i, :, 0] + data[i, :, 1],
        #     alpha=0.25,
        # )
        plt.errorbar(
            iterations,
            data[:, i, 0],
            data[:, i, 1],
            fmt=".",
            capsize=5,
            lw=1,
            label=f"{labels[i]} samples",
        )

    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    plt.axhline(y=0, lw=1, ls="--", c="black")
    plt.xticks(iterations)
    plt.xlabel("Training Iterations")
    plt.ylim(-1.1, 1.1)
    plt.ylabel("Expected Utility (P1)")
    plt.grid(alpha=0.25)
    plt.legend(title="MCTS Samples")
    plt.savefig(output_path + ".pdf")
    np.save(output_path, data)
    plt.close()


def plot_results(working_dir: str, num_samples_list: list[int], output_path: str):
    checkpoints = [
        int(x)
        for x in os.listdir(working_dir)
        if os.path.isdir(os.path.join(working_dir, x))
    ]
    checkpoints.sort()
    data = np.empty((len(checkpoints), len(num_samples_list), 2))

    for idx_1, checkpoint in enumerate(checkpoints):
        checkpoint_path = os.path.join(working_dir, str(checkpoint))

        for idx_2, num_samples in enumerate(num_samples_list):
            log_files = [
                log_file
                for log_file in os.listdir(
                    os.path.join(checkpoint_path, str(num_samples))
                )
                if log_file.endswith(".json")
            ]
            outcomes = []

            for log_file in log_files:
                file_path = os.path.join(checkpoint_path, str(num_samples), log_file)
                outcomes.append(get_outcome(file_path))
            mean = np.mean(outcomes)
            err = 1.96 * (np.std(outcomes) / np.sqrt(len(outcomes)))

            data[idx_1][idx_2] = np.array([mean, err])

    generate_plot(checkpoints, num_samples_list, data, output_path)


def aggregate_plot(dir: str, player: str, out: str):
    dir_path = pathlib.Path(dir)
    dirs = os.listdir(dir_path)
    data_list = []
    labels = [64, 128, 256]
    for sub in dirs:
        if not sub in [".DS_Store"]:
            data_path = os.path.join(dir_path, sub + f"/plots/player_{player}.npz.npy")
            d = np.load(data_path)
            print(data_path, d.shape)
            data_list.append(np.load(data_path))
    data_agg = np.stack(data_list, axis=0)
    data = np.mean(data_agg, axis=0)
    cerr = 1.96 * (np.std(data_agg, axis=0) / (np.sqrt(data_agg.shape[0])))
    # iterations = np.linspace(0, 30000, data.shape[0])
    iterations = np.linspace(0, 30000, 7)
    # inds = [0, 5, 10, 15, 20, 25, 30]

    fig, ax = plt.subplots()
    ax.set_aspect("auto")
    fig.set_figwidth(3)
    # fig.set_figwidth(3.75)

    for i in range(data.shape[1]):
        ax.errorbar(
            iterations,
            data[:, i, 0],
            cerr[:, i, 0],
            fmt=".",
            capsize=5,
            lw=1,
            label=f"{labels[i]} samples",
        )

    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.axhline(y=0, lw=1, ls="--", c="black")
    ax.set_xticks(iterations)
    ax.set_xlabel("Training Iterations")
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Expected Utility (P1)")
    ax.grid(alpha=0.25)
    ax.legend(title="MCTS Samples")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"bd_cc_6_6_6_{player}.pdf"), backend="pdf")


if __name__ == "__main__":
    dir: str = "/Users/zaheen/Documents/exps/3_11_26/bd_cc_6_6_6/"
    out: str = (
        "/Users/zaheen/Documents/notes/projects/chinese_checkers/experiments/imgs/03_11_26"
    )
    aggregate_plot(dir, "one", out)
    aggregate_plot(dir, "two", out)
