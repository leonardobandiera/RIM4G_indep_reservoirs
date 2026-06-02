from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.special import gammaln

sns.set_theme(style="whitegrid", context="talk")

base_path = Path("~/RIM4G/RIM4G_indep_reservoirs/uncoupled").expanduser()

# -------------------------
# j mapping (spin_number -> j)
# -------------------------
j_values = {
    4: 1, 8: 1, 16: 4, 32: 8, 64: 16,
    128: 16, 256: 16, 512: 16, 1024: 16, 2048: 32
}

# -------------------------
# SAFE CONVERTER
# -------------------------
def to_float(x):
    arr = np.asarray(x)

    if arr.dtype.type is np.str_ or arr.dtype.type is np.object_:
        try:
            return float(arr)
        except:
            return float(arr[-1])

    if arr.ndim > 1:
        arr = arr[:, -1]

    arr = arr.ravel()
    return float(arr[-1])

# -------------------------
# STORAGE
# -------------------------
plot_data = {
    "bias": {
        "spin_number": [],
        "entropy": [],
        "unc_mean": [],
        "unc_std": [],
    },
    "no_bias": {
        "spin_number": [],
        "entropy": [],
        "unc_mean": [],
        "unc_std": [],
    },
}

# -------------------------
# MAIN LOOP
# -------------------------
for blocks_dir in sorted(base_path.glob("*_blocks"),
                         key=lambda p: int(p.name.split("_")[0])):

    n_blocks = int(blocks_dir.name.split("_")[0])
    spin_number = int(4096 / n_blocks)

    if spin_number not in j_values:
        continue

    j = j_values[spin_number]

    # -------------------------
    # ENTROPY
    # -------------------------
    M = spin_number * (spin_number - 1) // 2
    k = j * spin_number

    log_binom = (
        gammaln(M + 1)
        - gammaln(k + 1)
        - gammaln(M - k + 1)
    )

    entropy = n_blocks * (log_binom + k * np.log(2))

    # -------------------------
    # COLLECT UNC (3 seeds)
    # -------------------------
    data_dict = {
        "bias": [],
        "no_bias": []
    }

    for f in blocks_dir.rglob("uncoupled_features.txt"):

        path_str = str(f)
        match = re.search(r"bias_(\d*\.?\d+)", path_str)

        if match is None:
            continue

        bias_value = float(match.group(1))
        key = "bias" if bias_value != 0 else "no_bias"

        try:
            raw = np.loadtxt(f)
            val = to_float(raw)

            if np.isnan(val):
                continue

            data_dict[key].append(val)

        except Exception as e:
            print(f"Skip {f}: {e}")
            continue

    # -------------------------
    # AGGREGATE OVER SEEDS
    # -------------------------
    for key in ["bias", "no_bias"]:

        vals = np.array(data_dict[key])

        if len(vals) == 0:
            continue

        plot_data[key]["spin_number"].append(spin_number)
        plot_data[key]["entropy"].append(entropy)
        plot_data[key]["unc_mean"].append(np.mean(vals))
        plot_data[key]["unc_std"].append(np.std(vals))

# -------------------------
# SORTING
# -------------------------
for key in ["bias", "no_bias"]:
    order = np.argsort(plot_data[key]["spin_number"])
    for field in plot_data[key]:
        plot_data[key][field] = np.array(plot_data[key][field])[order]

# -------------------------
# PLOT (1x2 + double y axis)
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True)
plt.subplots_adjust(wspace=0.5)

titles = {"bias": "With input features", "no_bias": "Topology-only"}

for ax, key in zip(axes, ["bias", "no_bias"]):

    ax2 = ax.twinx()

    x = np.array(plot_data[key]["spin_number"])

    # -------------------------
    # ENTROPY (LEFT AXIS)
    # -------------------------
    sns.lineplot(
        x=x,
        y=plot_data[key]["entropy"],
        marker="D",
        color="tab:blue",
        ax=ax,
        label="Entropy",
        legend=False
    )

    ax.set_ylabel("Entropy", color="tab:blue")
    ax.tick_params(axis='y', labelcolor="black")

    # -------------------------
    # UNCOUPLED (RIGHT AXIS)
    # -------------------------
    unc = np.array(plot_data[key]["unc_mean"])
    unc_std = np.array(plot_data[key]["unc_std"])

    ax2.errorbar(
        x,
        unc,
        yerr=unc_std,
        fmt="s-",
        color="tab:orange",
        capsize=4
    )

    ax2.set_ylabel("Uncoupled features", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="black")

    # -------------------------
    # X AXIS: LOG2
    # -------------------------
    ax.set_xscale("log", base=2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.tick_params(axis='x', rotation=45)

    ax.set_xlabel("Number of spins per block")

    ax.set_title(titles[key], color="black")
    ax.grid(True, alpha=0.5)
    ax2.grid(False)

    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0,0))

plt.savefig("entropy_uncoupled_good.pdf")
plt.tight_layout()
plt.show()
