from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

sns.set_theme(style="whitegrid", context="talk")

base_path = Path(
    "~/RIM4G/RIM4G_indep_reservoirs/results"
).expanduser()

datasets = ["Cora", "PubMed", "CiteSeer"]

rows = []

for dataset in datasets:

    dataset_path = base_path / dataset

    # cartelle tipo 8_blocks, 16_blocks, 32_blocks...
    for blocks_dir in sorted(
        dataset_path.glob("*_blocks"),
        key=lambda p: int(p.name.split("_")[0])
    ):

        n_blocks = int(blocks_dir.name.split("_")[0])

        spin_number = 4096 / n_blocks

        data_dict = {
            "bias": {
                "acc": [],
                "std": []
            },
            "no_bias":{
                "acc": [],
                "std": []
            }
        }

        for acc_file in blocks_dir.rglob("ridge_test_accuracy.txt"):

            path_str = str(acc_file)

            is_bias = bool(
                re.search(
                    r"\d*\.?\d+bias|bias\d*\.?\d+",
                    path_str
                )
            )

            try:

                acc_data = np.loadtxt(acc_file)
                acc_last = np.atleast_1d(acc_data)[-1]

                std_file = acc_file.parent / "std_test.txt"
                std_data = np.loadtxt(std_file)
                std_last = np.atleast_1d(std_data)[-1]

                key = "bias" if is_bias else "no_bias"

                data_dict[key]["acc"].append(acc_last)
                data_dict[key]["std"].append(std_last)

            except Exception as e:
                print(f"Errore con {acc_file}: {e}")

        for key in ["bias", "no_bias"]:

            accuracies = np.array(data_dict[key]["acc"])
            std_values = np.array(data_dict[key]["std"])

            if len(accuracies) == 0:
                continue

            mean_acc = np.mean(accuracies)

            # varianza seed
            variance_acc = np.mean(
                (accuracies - mean_acc)**2
            )

            # varianza fold
            variance_std = np.mean(std_values**2)

            total_std = np.sqrt(variance_acc + variance_std)

            rows.append({
                "dataset": dataset,
                "Number of spins per block": spin_number, 
                "Test accuracy": mean_acc,
                "total_std": total_std,
                "type": key
            })

df = pd.DataFrame(rows)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 16),
    sharex=True
)

all_x = sorted(df["Number of spins per block"].unique())

color_bias = "tab:orange"
color_no_bias = "tab:blue"

for ax, dataset in zip(axes, datasets):

    df_dataset = (
        df[df["dataset"] == dataset]
        .sort_values("Number of spins per block")
    )

    # with input features
    df_bias = df_dataset[df_dataset["type"] == "bias"]

    sns.lineplot(
        data=df_bias,
        x="Number of spins per block",
        y="Test accuracy",
        marker="o",
        color=color_bias,
        linewidth=2.5,
        ax=ax,
        label="with input features"
    )

    ax.errorbar(
        df_bias["Number of spins per block"],
        df_bias["Test accuracy"],
        yerr=df_bias["total_std"],
        fmt="none",
        ecolor=color_bias,
        capsize=5,
        elinewidth=2.5,
        alpha=0.8
    )

    # topology only
    df_no_bias = df_dataset[df_dataset["type"] == "no_bias"]

    sns.lineplot(
        data=df_no_bias,
        x="Number of spins per block",
        y="Test accuracy",
        marker="s",
        color=color_no_bias,
        linewidth=2.5,
        ax=ax,
        label="topology only"
    )

    ax.errorbar(
        df_no_bias["Number of spins per block"],
        df_no_bias["Test accuracy"],
        yerr=df_no_bias["total_std"],
        fmt="none",
        ecolor=color_no_bias,
        capsize=5,
        elinewidth=2.5,
        alpha=0.8
    )

    ax.set_title(dataset)
    ax.set_ylabel("Test accuracy [%]")
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_x)
    ax.set_xticklabels(
        [str(int(x)) for x in all_x]
    )

    ax.legend()

axes[-1].set_xlabel("Number of spins per block")

plt.tight_layout()

plt.savefig(
    "plot_all_datasets.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
