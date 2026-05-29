from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

base_path = Path(
    "~/RIM4G/RIM4G_indep_reservoirs/results"
).expanduser()

datasets = ["cora", "pubmed", "citeseer"]

plt.figure(figsize=(8, 6))

for dataset in datasets:

    dataset_path = base_path / dataset

    blocks = []
    means = []
    total_stds = []

    # cartelle tipo 8_blocks, 16_blocks, 32_blocks...
    for blocks_dir in sorted(
        dataset_path.glob("*_blocks"),
        key=lambda p: int(p.name.split("_")[0])
    ):

        n_blocks = int(blocks_dir.name.split("_")[0])

        accuracy = []
        std_values = []

        # cerca tutti i ridge_test_accuracy.txt
        for acc_file in blocks_dir.rglob("ridge_test_accuracy.txt"):

            path_str = str(acc_file)

            # escludi path con pattern tipo:
            # 1bias, 0.5bias, 10bias, ma anche bias1, bias0.5, ecc.
            if re.search(r"\d*\.?\d+bias|bias\d*\.?\d+", path_str):
                continue

            try:
                # accuracy
                acc_data = np.loadtxt(acc_file)
                acc_last = np.atleast_1d(acc_data)[-1]

                # std associata
                std_file = acc_file.parent / "std_test.txt"
                std_data = np.loadtxt(std_file)
                std_last = np.atleast_1d(std_data)[-1]

                accuracies.append(acc_last)
                std_values.append(std_last)

            except Exception as e:
                print(f"Errore con {file_path}: {e}")

        if len(accuracies) > 0:

            accuracies = np.array(accuracies)
            sts_values = np.array(std_values

                                  
            mean_acc = np.mean(accuracies)

            # calcolo di std complessiva
            # std dovuta ai seed:
            variance_acc = np.mean((accuracies - mean_acc) ** 2)
            # std dovuta ai fold:
            variance_std = np.mean(std_values ** 2)
            
            std_tot = np.sqrt(variance_acc + variance_std)

            blocks.append(n_blocks)
            means.append(mean_acc)
            total_stds.append(total_std)

            print(f"\n{dataset} - {n_blocks} blocks")
            print(f"accuracies = {accuracies}")
            print(f"std_values = {std_values}")
            print(f"mean       = {mean_acc:.4f}")
            print(f"total std  = {total_std:.4f}")

    # plot dataset corrente
    plt.errorbar(
        blocks,
        means,
        yerr=total_stds,
        fmt='o-',
        capsize=5,
        label=dataset
    )

plt.xlabel("Number of blocks")
plt.ylabel("Test accuracy [%]")

plt.xscale('log', base=2)

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
