from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

# cartella principale results/
base_path = Path(
    "~/RIM4G/RIM4G_indep_reservoirs/results"
).expanduser()

# dataset da confrontare
datasets = ["cora", "pubmed", "citeseer"]

plt.figure(figsize=(8, 6))

for dataset in datasets:

    dataset_path = base_path / dataset

    blocks = []
    means = []
    stds = []

    # cartelle tipo 8_blocks, 16_blocks, 32_blocks...
    for blocks_dir in sorted(
        dataset_path.glob("*_blocks"),
        key=lambda p: int(p.name.split("_")[0])
    ):

        n_blocks = int(blocks_dir.name.split("_")[0])

        values = []

        # cerca tutti i ridge_test_accuracy.txt
        for file_path in blocks_dir.rglob("ridge_test_accuracy.txt"):

            path_str = str(file_path)

            # escludi path con pattern tipo:
            # 1bias, 0.5bias, 10bias, ecc.
            if re.search(r"\d*\.?\d+bias", path_str):
                continue

            try:
                data = np.loadtxt(file_path)

                # prende l'ultimo valore
                last_value = np.atleast_1d(data)[-1]

                values.append(last_value)

            except Exception as e:
                print(f"Errore con {file_path}: {e}")

        if len(values) > 0:

            mean = np.mean(values)
            std = np.std(values)

            blocks.append(n_blocks)
            means.append(mean)
            stds.append(std)

            print(f"\n{dataset} - {n_blocks} blocks")
            print(f"values = {values}")
            print(f"mean   = {mean:.4f}")
            print(f"std    = {std:.4f}")

    # plot dataset corrente
    plt.errorbar(
        blocks,
        means,
        yerr=stds,
        fmt='o-',
        capsize=5,
        label=dataset
    )

# -------------------
# grafico finale
# -------------------

plt.xlabel("Number of blocks")
plt.ylabel("Test accuracy [%]")

plt.xscale('log', base=2)

# tick asse x
all_blocks = sorted({
    int(p.name.split("_")[0])
    for dataset in datasets
    for p in (base_path / dataset).glob("*_blocks")
})

plt.xticks(all_blocks, all_blocks)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig("all_datasets_accuracy.png", dpi=300)

plt.show()
