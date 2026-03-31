import json
import os

import matplotlib.pyplot as plt
import numpy as np

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


def generate_plot(
    iterations: list[int], labels: list[int], data: np.ndarray, output_path: str
):
    plt.rcParams.update(
        {"pgf.texsystem": "lualatex", "text.usetex": True, "font.family": "serif"}
    )
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

    plt.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    plt.axhline(y=0, lw=1, ls="--", c="black")
    plt.xticks(iterations)
    plt.xlabel("Training Iterations")
    plt.ylim(-1.1, 1.1)
    plt.ylabel("Expected Utility (P1)")
    plt.grid(alpha=0.25)
    plt.legend(title="MCTS Samples")
    plt.savefig(output_path + ".pdf")
    plt.close()


def plot_results(working_dir: str, num_samples_list: list[int], output_path: str):
    checkpoints = [
        int(x)
        for x in os.listdir(working_dir)
        if os.path.isdir(os.path.join(working_dir, x))
    ]
    checkpoints.sort()
    print(checkpoints)
    print(num_samples_list)
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
