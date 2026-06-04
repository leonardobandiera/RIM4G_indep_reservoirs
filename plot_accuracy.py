from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import ttest_ind_from_stats

sns.set_theme(style="whitegrid", context="talk")

base_path = Path(
    "~/RIM4G/RIM4G_indep_reservoirs/results"
).expanduser()

datasets = ["Cora", "PubMed", "CiteSeer"]

manual_values = {
    "Cora": {
        "bias": {"y": 86.19, "err": 1.51},
        "no_bias": {"y": 84.2, "err": 1.1},
    },
    "PubMed": {
        "bias": {"y": 87.41, "err": 0.38},
        "no_bias": {"y": 81.7, "err": 0.6},
    },
    "CiteSeer": {
        "bias": {"y": 74.27, "err": 1.78},
        "no_bias": {"y": 64.2, "err": 4.9},
    }
}

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
                "type": key,
                "raw_acc": accuracies
            })

#### test statistico ###
df = pd.DataFrame(rows)

n_folds = 10

baseline_stats = {}

for dataset in datasets:
    for typ in ["bias", "no_bias"]:
        baseline_stats[(dataset, typ)] = {"mean": manual_values[dataset][typ]["y"], "std": manual_values[dataset][typ]["err"]}




p_values = []
for _, row in df.iterrows():
    k = row["Number of spins per block"]

    if k == 4096:
        p_values.append(np.nan)
        continue
    dataset = row["dataset"]
    typ = row["type"]

    mean1 = row["Test accuracy"]
    std1 = row["total_std"]

    mean2 = baseline_stats[(dataset, typ)]["mean"]
    std2 = baseline_stats[(dataset, typ)]["std"]

    _, p = ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=n_folds, mean2=mean2, std2=std2, nobs2=n_folds, equal_var=False)

    p_values.append(p)

df["p_value"] = p_values
df["p_value"] = df["p_value"].map(lambda x: f"'{x:.10f}")
df.to_csv("reservoir_pvalues_results.csv", index=False)

### plot ###


fig, axes = plt.subplots(
    3,
    2,
    figsize=(14, 16),
    sharex=True
)

all_x = sorted(df["Number of spins per block"].unique())
all_x = sorted(set(all_x + [4096]))

color_bias = "tab:orange"
color_no_bias = "tab:blue"
for i, dataset in enumerate(datasets):

    df_dataset = (
        df[df["dataset"] == dataset]
        .sort_values("Number of spins per block")
    )

    # no bias
    ax = axes[i, 0]
    df_no_bias = df_dataset[df_dataset["type"] == "no_bias"]

    sns.lineplot(
        data=df_no_bias,
        x="Number of spins per block",
        y="Test accuracy",
        marker="s",
        color=color_no_bias,
        linewidth=2.5,
        ax=ax
    )

    ax.errorbar(
        df_no_bias["Number of spins per block"],
        df_no_bias["Test accuracy"],
        yerr=df_no_bias["total_std"],
        fmt="none",
        ecolor=color_no_bias,
        capsize=5,
        elinewidth=1.5,
        alpha=0.8
    )

    y_ref = manual_values[dataset]["no_bias"]["y"]
    err_ref = manual_values[dataset]["no_bias"]["err"]

    last_x = df_no_bias["Number of spins per block"].max()
    last_y = df_no_bias.loc[df_no_bias["Number of spins per block"].idxmax(), "Test accuracy"]
    ax.plot([last_x, 4096], [last_y, y_ref], color=color_no_bias, linewidth=2.5)

    ax.errorbar(4096, y_ref, yerr=err_ref, fmt="D", color="tab:green", capsize=6, markersize=8)
    ax.axhline(y_ref, color="tab:green", linestyle="--", linewidth=1.8, alpha=0.7)

    ax.set_title(f"{dataset} — topology-only")
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_x)
    ax.set_xlim(left=2)
    ax.set_ylabel("Test accuracy [%]")
    ax.set_xticklabels([str(int(x)) for x in all_x])
    ax.yaxis.set_major_formatter(
    FormatStrFormatter('%.1f')
    )
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis='x', labelbottom=True)

    # input features
    ax = axes[i, 1]
    df_bias = df_dataset[df_dataset["type"] == "bias"]

    sns.lineplot(
        data=df_bias,
        x="Number of spins per block",
        y="Test accuracy",
        marker="o",
        color=color_bias,
        linewidth=2.5,
        ax=ax
    )

    ax.errorbar(
        df_bias["Number of spins per block"],
        df_bias["Test accuracy"],
        yerr=df_bias["total_std"],
        fmt="none",
        ecolor=color_bias,
        capsize=5,
        elinewidth=1.5,
        alpha=0.8
    )

    y_ref = manual_values[dataset]["bias"]["y"]
    err_ref = manual_values[dataset]["bias"]["err"]

    last_x = df_bias["Number of spins per block"].max()
    last_y = df_bias.loc[df_bias["Number of spins per block"].idxmax(), "Test accuracy"]
    ax.plot([last_x, 4096], [last_y, y_ref], color=color_bias, linewidth=2.5)
    ax.errorbar(4096, y_ref, yerr=err_ref, fmt="D", color="tab:green", capsize=6, markersize=8)
    ax.axhline(y_ref, color="tab:green", linestyle="--", linewidth=1.8, alpha=0.7)
    ax.set_title(f"{dataset} — with input features")
    ax.set_xscale("log", base=2)
    ax.set_xlim(left=2)
    ax.set_ylabel("Test accuracy [%]")
    ax.set_xticks(all_x)
    ax.yaxis.set_major_formatter(
    FormatStrFormatter('%.1f')
    )
    ax.set_xticklabels([str(int(x)) for x in all_x])
    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(axis='x', labelbottom=True)

axes[-1, 0].set_xlabel("Number of spins per block")
axes[-1, -1].set_xlabel("Number of spins per block")

plt.tight_layout()

plt.savefig(
    "plot_all_datasets.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
