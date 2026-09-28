import torch
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = r"D:/Thermal surrogate project/train_raw_graphs.pt"
OUTPUT_PATH = r"D:/Thermal surrogate project/high_temp_graphs_70pct.pt"

LOW_TEMP = 450.0
HIGH_TEMP = 500.0
MIN_RATIO = 0.70


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("Loading raw training dataset...")
print("=" * 70)

dataset = torch.load(
    DATASET_PATH,
    map_location="cpu",
    weights_only=False
)

print(f"Total graphs in dataset: {len(dataset)}")


# ============================================================
# EXTRACT HIGH-TEMPERATURE GRAPHS
# ============================================================

selected_graphs = []

ratios = []

for graph_idx, graph in enumerate(dataset):

    # --------------------------------------------------------
    # Get target temperature
    # --------------------------------------------------------
    target = graph.y

    # Make sure target is 1D
    target = target.view(-1)

    total_nodes = target.numel()

    if total_nodes == 0:
        continue

    # --------------------------------------------------------
    # Nodes inside 450-500 K
    # --------------------------------------------------------
    high_temp_mask = (
        (target >= LOW_TEMP) &
        (target <= HIGH_TEMP)
    )

    high_temp_nodes = high_temp_mask.sum().item()

    # --------------------------------------------------------
    # Percentage of nodes in 450-500 K
    # --------------------------------------------------------
    high_temp_ratio = high_temp_nodes / total_nodes

    ratios.append(high_temp_ratio)

    # --------------------------------------------------------
    # Select graph if >= 70%
    # --------------------------------------------------------
    if high_temp_ratio >= MIN_RATIO:

        selected_graphs.append(graph)

        print(
            f"Graph {graph_idx:5d} | "
            f"Nodes: {total_nodes:5d} | "
            f"450-500K Nodes: {high_temp_nodes:5d} | "
            f"Ratio: {high_temp_ratio * 100:6.2f}%"
        )


# ============================================================
# SAVE SELECTED GRAPHS
# ============================================================

torch.save(
    selected_graphs,
    OUTPUT_PATH
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("HIGH-TEMPERATURE GRAPH EXTRACTION COMPLETE")
print("=" * 70)

print(f"Original graphs      : {len(dataset)}")
print(f"Selected graphs      : {len(selected_graphs)}")

if len(dataset) > 0:
    percentage = len(selected_graphs) / len(dataset) * 100
    print(f"Dataset percentage   : {percentage:.2f}%")

print(f"Temperature range    : {LOW_TEMP} - {HIGH_TEMP} K")
print(f"Minimum node ratio   : {MIN_RATIO * 100:.1f}%")

print(f"\nSaved to:")
print(OUTPUT_PATH)

print("=" * 70)