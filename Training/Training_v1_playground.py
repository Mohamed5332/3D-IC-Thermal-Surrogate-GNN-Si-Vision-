import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


# imports
import json
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as geometric_nn
from torch_geometric.loader import DataLoader
from joblib import load, dump



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


loaders = load(filename="loaders")

"""
HARD_DATASET_PATH = r"D:/Thermal surrogate project/Datasets/hard_graphs_subset.pt"
HIGH_TEMP_DATASET_PATH = r"D:/Thermal surrogate project/Datasets/high_temp_graphs_70pct.pt"
MID_HIGH_TEMP_DATASET_PATH = r"D:/Thermal surrogate project/Datasets/400_450_temp_graphs_70pct.pt"


new_real_400_500k_datasetv1 = torch.load(
    f= 'D:/Thermal surrogate project/Datasets/thermal_graph_phase2d_mate_400_500K_ONLY_v2schema.pt', 
    weights_only = False
)

new_real_400_500k_datasetv2 = torch.load(
    f= 'D:/Thermal surrogate project/Datasets/thermal_graph_phase2d_mate_run400_500_v2schema.pt', 
    weights_only = False
)

new_real_450_500k_datasetv2 = torch.load(
    f = 'D:/Thermal surrogate project/Datasets/thermal_graph_filtered_450_500K_v2schema.pt', 
    weights_only = False
)

hard_graphs_dataset = torch.load(
    f = HARD_DATASET_PATH,
    weights_only = False
)

above450_dataset = torch.load(
    f=HIGH_TEMP_DATASET_PATH,
    weights_only=False
)

above400_450_dataset = torch.load(
    f=MID_HIGH_TEMP_DATASET_PATH,
    weights_only=False
)


original_train_dataset = loaders["train"].dataset

from Preprocessing.Train_data_processin import Train_Nodes_data_pipleline , Train_Edges_data_pipleline , Train_Global_data_pipleline_new , Train_normalize_temp
hard_graphs_dataset_processed = Train_Nodes_data_pipleline(
    train_data=hard_graphs_dataset
)
hard_graphs_dataset_processed = Train_Edges_data_pipleline(
    train_data=hard_graphs_dataset_processed
)
hard_graphs_dataset_processed = Train_Global_data_pipleline_new(
    train_data=hard_graphs_dataset_processed
)
hard_graphs_dataset_processed = Train_normalize_temp(
    train_data=hard_graphs_dataset_processed
)

above450_dataset_processed = Train_Nodes_data_pipleline(
    train_data=above450_dataset
)
above450_dataset_processed = Train_Edges_data_pipleline(
    train_data=above450_dataset_processed
)
above450_dataset_processed = Train_Global_data_pipleline_new(
    train_data=above450_dataset_processed
)
above450_dataset_processed = Train_normalize_temp(
    train_data=above450_dataset_processed
)

above400_dataset_processed = Train_Nodes_data_pipleline(
    train_data=above400_450_dataset
)
above400_dataset_processed = Train_Edges_data_pipleline(
    train_data=above400_dataset_processed
)
above400_dataset_processed = Train_Global_data_pipleline_new(
    train_data=above400_dataset_processed
)
above400_dataset_processed = Train_normalize_temp(
    train_data=above400_dataset_processed
)

new_real_400_500k_datasetv1_processed = Train_Nodes_data_pipleline(train_data= new_real_400_500k_datasetv1)
new_real_400_500k_datasetv1_processed = Train_Edges_data_pipleline(train_data= new_real_400_500k_datasetv1_processed)
new_real_400_500k_datasetv1_processed = Train_Global_data_pipleline_new(train_data= new_real_400_500k_datasetv1_processed)
new_real_400_500k_datasetv1_processed = Train_normalize_temp(train_data= new_real_400_500k_datasetv1_processed)


new_real_400_500k_datasetv2_processed = Train_Nodes_data_pipleline(train_data = new_real_400_500k_datasetv2)
new_real_400_500k_datasetv2_processed = Train_Edges_data_pipleline(train_data = new_real_400_500k_datasetv2_processed)
new_real_400_500k_datasetv2_processed = Train_Global_data_pipleline_new(train_data = new_real_400_500k_datasetv2_processed)
new_real_400_500k_datasetv2_processed = Train_normalize_temp(train_data = new_real_400_500k_datasetv2_processed)


new_real_450_500k_datasetv2_processed = Train_Nodes_data_pipleline(train_data = new_real_450_500k_datasetv2)
new_real_450_500k_datasetv2_processed = Train_Edges_data_pipleline(train_data = new_real_450_500k_datasetv2_processed)
new_real_450_500k_datasetv2_processed = Train_Global_data_pipleline_new(train_data = new_real_450_500k_datasetv2_processed)
new_real_450_500k_datasetv2_processed = Train_normalize_temp(train_data = new_real_450_500k_datasetv2_processed)

# Combine datasets for oversampling...
Fine_tuning_data = (list(original_train_dataset) + list(new_real_400_500k_datasetv2_processed) + list(new_real_400_500k_datasetv1_processed) + list(new_real_450_500k_datasetv2_processed))
"""

original_train_dataset = loaders["train"].dataset
Fine_tuning_data = list(original_train_dataset)


# removing graphs that contains temp outside our essential range...
def remove_graphs_outside_temperature_range(dataset, min_temp=300.0, max_temp=500.0):
    filtered_dataset = []
    removed_count = 0

    for graph in dataset:

        valid_temps = graph.y

        if ((valid_temps < min_temp) | (valid_temps > max_temp)).any():
            removed_count += 1
            continue

        filtered_dataset.append(graph)

    print(f"Original graphs : {len(dataset)}")
    print(f"Removed graphs  : {removed_count}")
    print(f"Remaining graphs: {len(filtered_dataset)}")

    return filtered_dataset



Fine_tuning_data = remove_graphs_outside_temperature_range(dataset= Fine_tuning_data , min_temp= 300 , max_temp= 500)


ULTRA_BATCH_SIZE = 16
# Build loader...
Fine_tuning_loader = DataLoader(
    dataset=Fine_tuning_data,
    batch_size=ULTRA_BATCH_SIZE,
    shuffle=True
)

# Validation remains the ORIGINAL validation loader.
validation_loader = loaders["val"]




Edges_scalers = load(filename="scallers_Edege_data")
std_resistance = Edges_scalers["std_resistance"]

Nodes_scalers = load(filename="scallers_Node_data")
std_power_density    = Nodes_scalers["std_power_density"]
std_area             = Nodes_scalers["std_area"]
std_kappa            = Nodes_scalers["std_kappa"]
std_exposed_area_mm2 = Nodes_scalers.get("std_exposed_area_mm2")

Global_scalers = load(filename="scallers_global_data")
std_ambient_temp_K = Global_scalers["std_ambient_temp_K"]

normalized_temp_scaler = load(filename="normalized_temp_scaller")
std_target_temp = normalized_temp_scaler["std_target_temp"]


# To get mean_ and scale_ ...
def get_scaler_value(scaler, attribute, device):
    """Safely extract a scalar from an sklearn-like scaler."""
    value = getattr(scaler, attribute)
    value = np.asarray(value).reshape(-1)[0]
    return torch.tensor(float(value), dtype=torch.float32, device=device)


target_temp_mean  = get_scaler_value(std_target_temp, "mean_", device)
target_temp_scale = get_scaler_value(std_target_temp, "scale_", device)


CONFIG = {
    "in_features": 18,
    "in_features_edge": 11,
    "global_features": 21,
    "hidden_features": 64,
    "edge_mlp_hidden": 32,
    "num_layers": 4,
    "gat_heads": 2,

    "temperature_ranges": [
        (300.0, 350.0),
        (350.0, 400.0),
        (400.0, 450.0),
        (450.0, 500.0),
    ],

    # Width (K) of the soft-transition zone centered on each internal
    # boundary (350, 400, 450). A node this far or more from every
    # boundary belongs 100% to a single range, exactly like before.
    # A node closer than this to a boundary is blended between the two
    # adjacent ranges. Set to 0.0 to fully recover the old hard-cut
    # behaviour if you ever want to A/B compare.
    "boundary_margin_K": 0.0,

    "real_space_loss_weight": 1,
    "real_space_loss_scale_K": 30.0,

    # [CHANGE] halved from 0.1 -> 0.05. In your run, every stability
    # regression (rising train/val gap, ranges reversing bias direction)
    # started exactly when lambda_physics hit its full value at epoch 15.
    # A smaller final physics weight still enforces global energy
    # conservation but leaves more room for the range-balanced ML/real
    # losses to control per-range accuracy without the two objectives
    # fighting each other as hard.
    "lambda_physics": 0.05,
    "physics_warmup_epochs": 20,

    "lr": 1e-3,
    # [CHANGE] increased from 1e-4 -> 3e-4. Your train/val gap grew
    # monotonically (1.86% -> 6.67% -> 14.11% -> 21.47%) with train loss
    # still falling while val loss stalled/worsened -- the textbook
    # signature of the model starting to memorize the training graphs.
    # Stronger weight decay constrains parameter magnitudes and reduces
    # that capacity to overfit.
    "weight_decay": 3e-4,
    "grad_clip": 35,
    "epochs": 200,

    "scheduler_factor": 0.5,
    "scheduler_patience": 1,

    "early_stopping_patience": 12,

    # [NEW] Hard safety net independent of RMSE-based early stopping.
    # RMSE-based early stopping alone did not catch your overfitting
    # episode because val RMSE was still (barely) improving even as the
    # gap exploded to +21%. This checks the gap directly, every epoch,
    # once physics is past warmup (early epochs naturally show a larger
    # gap while the physics term is still ramping up and is not a sign
    # of overfitting).
    "max_safe_train_val_gap_pct": 25.0,

    "use_lognormal_correction": False,

    "diagnostic_bin_size_K": 10.0,

    "checkpoint_path": "Model_v8_playground.pt",
    "results_path": "Model_v8_playground.json",
    "training_mode": "from_scratch",
    "batch_size": 8,
    "gap_guard_buffer_epochs": 15
}

TEMPERATURE_BINS = CONFIG["temperature_ranges"]
BOUNDARY_MARGIN_K = CONFIG["boundary_margin_K"]


# build ranges 300 - 310 , 310 - 310 till 500...
DIAGNOSTIC_BINS = [
    (300.0 + i * CONFIG["diagnostic_bin_size_K"],
     300.0 + (i + 1) * CONFIG["diagnostic_bin_size_K"])
    for i in range(int((500.0 - 300.0) / CONFIG["diagnostic_bin_size_K"]))
]



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# determine each node belongs to which range with a weight for boundaries...
def soft_range_weights(target_kelvin , ranges , margin_K):
    """
    Returns a [num_ranges, N] tensor of weights in [0, 1].

    For each node, weights across all ranges sum to 1 (every node is
    fully accounted for exactly once), but instead of an all-or-nothing
    assignment, a node within `margin_K` of an internal boundary splits
    its weight linearly between the two neighbouring ranges.

    Example with margin_K=10, boundary at 400 K:
        T=390 -> range[350-400] weight 1.0,           range[400-450] weight 0.0
        T=395 -> range[350-400] weight 0.75,           range[400-450] weight 0.25
        T=400 -> range[350-400] weight 0.5,             range[400-450] weight 0.5
        T=405 -> range[350-400] weight 0.25,           range[400-450] weight 0.75
        T=410 -> range[350-400] weight 0.0,             range[400-450] weight 1.0
    """

    num_ranges = len(ranges)
    N = target_kelvin.shape[0]

    weights = torch.zeros(num_ranges, N, dtype=target_kelvin.dtype, device=target_kelvin.device)

    half = margin_K / 2.0

    for i, (low, high) in enumerate(ranges):
        # Base membership: fully inside this range's core (away from any boundary).
        core_low = low + (half if i > 0 else -1e9)
        core_high = high - (half if i < num_ranges - 1 else -1e9)

        in_core = (target_kelvin >= core_low) & (target_kelvin < core_high)
        weights[i] = torch.where(in_core, torch.ones_like(target_kelvin), weights[i])

        # Left transition zone (shared with range i-1), only if not the first range.
        if i > 0:
            left_lo = low - half
            left_hi = low + half
            in_left_zone = (target_kelvin >= left_lo) & (target_kelvin < left_hi)
            # linear ramp: 0 at left_lo -> 1 at left_hi
            ramp = (target_kelvin - left_lo) / max(margin_K, 1e-8)
            weights[i] = torch.where(in_left_zone, ramp, weights[i])
            weights[i - 1] = torch.where(in_left_zone, 1.0 - ramp, weights[i - 1])

    # Clip for numerical safety, then renormalize so every node's weights sum to 1.
    weights = weights.clamp(min=0.0, max=1.0)
    totals = weights.sum(dim=0, keepdim=True).clamp(min=1e-8)
    weights = weights / totals

    return weights



# Check that number of nodes in x and normalized_temp and y are equal to ensure correctness of data preprocessing...
def assert_all_nodes_are_targets(loader):
    batch = next(iter(loader))
    n_nodes = batch.x.size(0)
    n_target = batch.T_real_normalized.size(0)
    y_nodes = batch.y.numel()

    print("\n" + "=" * 40)
    print("ALL-NODES TARGET CHECK")
    print("=" * 40)
    print(f"Nodes             : {n_nodes}")
    print(f"T_real_normalized : {n_target}")
    print(f"y                 : {y_nodes}")

    if n_target != n_nodes:
        raise RuntimeError("T_real_normalized does not contain one target for every node.")
    if y_nodes != n_nodes:
        raise RuntimeError("batch.y does not contain one target for every node.")

    print("OK: every node is supervised.")


# Check that the required attributes for building physics loss exist already as a features...
def assert_physics_fields_exist(loader):
    batch = next(iter(loader))
    required_fields = ["physics_node_power_W", "G_amb_W_K", "pinn_global_energy_eligible"]

    print("\n" + "=" * 40)
    print("PHYSICS FIELD CHECK")
    print("=" * 40)

    for field in required_fields:
        if not hasattr(batch, field):
            raise RuntimeError(f"Missing physics field: {field}")
        print(f"{field:<40s}: OK")



# Calculate Number of Nodes in each range and check if nodes outside the range (300-500)k...
def compute_temperature_distribution(loader):
    counts = torch.zeros(len(TEMPERATURE_BINS), dtype=torch.long)
    out_of_range = 0

    for batch in loader:
        temperature = batch.y.detach().cpu().reshape(-1)
        in_any = torch.zeros(temperature.shape, dtype=torch.bool)

        for i, (low, high) in enumerate(TEMPERATURE_BINS):
            if i == len(TEMPERATURE_BINS) - 1:
                mask = (temperature >= low) & (temperature <= high)
            else:
                mask = (temperature >= low) & (temperature < high)
            counts[i] += mask.sum()
            in_any |= mask

        out_of_range += int((~in_any).sum())

    print("\n" + "=" * 40)
    print("TRAIN TEMPERATURE DISTRIBUTION")
    print("=" * 40)

    total = counts.sum().item()
    for i, (low, high) in enumerate(TEMPERATURE_BINS):
        count = counts[i].item()
        pct = 100.0 * count / max(total, 1)
        print(f"{low:.0f}-{high:.0f} K : {count:8d} nodes ({pct:6.2f}%)")

    print(f"Total in objective ranges: {total}")
    if out_of_range > 0:
        print(f"\nWARNING: {out_of_range} nodes are outside the 300-500 K objective.")

# Compute average loss per range (calc loss for each range RMSE)
class RangeBalancedMLLoss(nn.Module):
    def __init__(self, temperature_ranges, margin_K):
        super().__init__()
        self.ranges = temperature_ranges
        self.margin_K = margin_K

    def forward(self, prediction_normalized, target_normalized, target_kelvin):
        per_node_loss = (prediction_normalized - target_normalized) ** 2

        weights = soft_range_weights(target_kelvin, self.ranges, self.margin_K)

        range_losses = []
        for i in range(len(self.ranges)):
            w = weights[i] # [1,0,0,0]
            w_sum = w.sum()
            if w_sum > 1e-8:
                range_loss = (per_node_loss * w).sum() / w_sum
                range_losses.append(range_loss)

        if not range_losses:
            raise RuntimeError("No target temperatures fall inside the configured ranges.")

        return torch.stack(range_losses).mean()



# Huber loss...
class RangeBalancedRealSpaceLoss(nn.Module):
    def __init__(self, temperature_ranges, margin_K, scale_K=10.0):
        super().__init__()
        self.ranges = temperature_ranges
        self.margin_K = margin_K
        self.scale_K = scale_K

    def forward(self, prediction_kelvin, target_kelvin):
        error_scaled = (prediction_kelvin - target_kelvin) / self.scale_K
        per_node_loss = F.huber_loss(
            error_scaled, torch.zeros_like(error_scaled), reduction="none", delta=1.0
        )

        weights = soft_range_weights(target_kelvin, self.ranges, self.margin_K)

        range_losses = []
        for i in range(len(self.ranges)):
            w = weights[i]
            w_sum = w.sum()
            if w_sum > 1e-8:
                range_loss = (per_node_loss * w).sum() / w_sum
                range_losses.append(range_loss)

        if not range_losses:
            raise RuntimeError("No target temperatures fall inside the configured ranges.")

        return torch.stack(range_losses).mean()


# ============================================================
# GLOBAL PHYSICS LOSS  (unchanged logic, just cleaned references)
# ============================================================

def calculate_global_energy_loss(T_pred_kelvin, batch):
    T_pred_kelvin = T_pred_kelvin.reshape(-1)
    device_physics = T_pred_kelvin.device

    node_graph_id = batch.batch.to(device_physics)
    num_graphs = batch.num_graphs

    P_node = batch.physics_node_power_W.reshape(-1).to(device_physics)
    P_total = torch.zeros(num_graphs, dtype=T_pred_kelvin.dtype, device=device_physics)
    P_total.index_add_(0, node_graph_id, P_node)

    G_amb = batch.G_amb_W_K.reshape(-1).to(device_physics)

    T_amb_normalized = batch.global_features[:, 0]

    amb_scale = torch.as_tensor(
        np.asarray(std_ambient_temp_K.scale_).reshape(-1)[0],
        dtype=T_pred_kelvin.dtype, device=device_physics,
    )
    amb_mean = torch.as_tensor(
        np.asarray(std_ambient_temp_K.mean_).reshape(-1)[0],
        dtype=T_pred_kelvin.dtype, device=device_physics,
    )

    T_amb_real = T_amb_normalized * amb_scale + amb_mean
    T_amb_node = T_amb_real[node_graph_id]

    Q_ambient_node = G_amb * (T_pred_kelvin - T_amb_node)

    Q_ambient_total = torch.zeros(num_graphs, dtype=T_pred_kelvin.dtype, device=device_physics)
    Q_ambient_total.index_add_(0, node_graph_id, Q_ambient_node)

    residual_global = P_total - Q_ambient_total

    eligible = batch.pinn_global_energy_eligible.reshape(-1).bool().to(device_physics)

    if not eligible.any():
        return T_pred_kelvin.sum() * 0.0, residual_global.detach()

    normalized_residual = residual_global[eligible] / P_total[eligible].abs().clamp(min=1e-8)
    physics_loss = (normalized_residual ** 2).mean()

    return physics_loss, residual_global.detach()


# ============================================================
# MODEL  (unchanged architecture)
# ============================================================

class ML_only_model(nn.Module):
    def __init__(self, in_features, hidden_features, in_features_edge,
                 global_features, num_layers=4, edge_mlp_hidden=32, gat_heads=2):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_features, hidden_features),
        )

        self.nn_convs = nn.ModuleList()
        self.gat_convs = nn.ModuleList()
        self.merges = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            edge_mlp = nn.Sequential(
                nn.Linear(in_features_edge, edge_mlp_hidden),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(edge_mlp_hidden, hidden_features * hidden_features),
            )

            self.nn_convs.append(
                geometric_nn.NNConv(hidden_features, hidden_features, edge_mlp, aggr="add")
            )
            self.gat_convs.append(
                geometric_nn.GATv2Conv(
                    in_channels=hidden_features, out_channels=hidden_features,
                    heads=gat_heads, concat=False, edge_dim=in_features_edge,
                    add_self_loops=False,
                )
            )
            self.merges.append(nn.Linear(hidden_features * 2, hidden_features))
            self.norms.append(nn.LayerNorm(hidden_features))

        self.global_encoder = nn.Sequential(
            nn.Linear(global_features, hidden_features),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_features, hidden_features),
        )

        self.FC = nn.Sequential(
            nn.Linear(hidden_features * 2, hidden_features),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, x, edge_index, edge_attr, global_features):
        h = self.input_proj(x)

        for conv_nn, conv_gat, merge, norm in zip(
            self.nn_convs, self.gat_convs, self.merges, self.norms
        ):
            residual = h
            h_nn = conv_nn(h, edge_index, edge_attr)
            h_gat = conv_gat(h, edge_index, edge_attr)
            h = merge(torch.cat([h_nn, h_gat], dim=-1))
            h = F.leaky_relu(input = h , negative_slope= 0.01)
            h = h + residual
            h = norm(h)

        global_embedding = self.global_encoder(global_features)
        final_features = torch.cat([h, global_embedding], dim=-1)

        return self.FC(final_features).squeeze(-1)


# ============================================================
# DENORMALIZATION
# ============================================================

def normalized_to_kelvin(prediction_normalized):
    log_prediction = prediction_normalized * target_temp_scale + target_temp_mean
    return torch.exp(log_prediction)


def apply_jensen_correction(prediction_normalized, residual_log_var):
    """
    residual_log_var can be a scalar tensor (legacy, single global
    correction) OR a [num_ranges] tensor (new: per-range correction,
    see estimate_jensen_sigma_per_range). If per-range, the correction
    applied to each node uses the variance of the range it falls in
    (hard nearest-range lookup is fine here since this only affects the
    magnitude of a smooth multiplicative correction, not a training
    boundary discontinuity).
    """
    log_prediction = prediction_normalized * target_temp_scale + target_temp_mean

    if CONFIG["use_lognormal_correction"]:
        log_prediction = log_prediction + 0.5 * residual_log_var

    return torch.exp(log_prediction)


# ============================================================
# PER-RANGE JENSEN SIGMA ESTIMATION
# ============================================================

@torch.no_grad()
def estimate_jensen_sigma_per_range(model, loader):
    """
    Returns a tensor [num_ranges] with the log-space residual variance
    computed separately for each temperature range, plus a lookup
    tensor so validation can assign the correct per-node variance.
    """
    model.eval()

    all_residual_log = []
    all_targets_kelvin = []

    for batch in loader:
        batch = batch.to(device)

        prediction = model(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.global_features[batch.batch],
        )

        target = batch.T_real_normalized
        residual = (target - prediction) * target_temp_scale.detach()

        all_residual_log.append(residual.detach().cpu())
        all_targets_kelvin.append(batch.y.detach().cpu())

    all_residual_log = torch.cat(all_residual_log)
    all_targets_kelvin = torch.cat(all_targets_kelvin)

    per_range_var = torch.zeros(len(TEMPERATURE_BINS))

    for i, (low, high) in enumerate(TEMPERATURE_BINS):
        if i == len(TEMPERATURE_BINS) - 1:
            mask = (all_targets_kelvin >= low) & (all_targets_kelvin <= high)
        else:
            mask = (all_targets_kelvin >= low) & (all_targets_kelvin < high)

        if mask.any():
            per_range_var[i] = all_residual_log[mask].var(unbiased=False)
        else:
            per_range_var[i] = all_residual_log.var(unbiased=False)  # fallback

    return per_range_var.to(device)


def per_node_range_variance(target_kelvin, per_range_var):
    """
    Assigns each node the variance of the range its temperature falls
    into (hard lookup -- fine here, this only scales a smooth
    correction and does not create a training-time loss discontinuity).
    """
    result = torch.zeros_like(target_kelvin)
    for i, (low, high) in enumerate(TEMPERATURE_BINS):
        if i == len(TEMPERATURE_BINS) - 1:
            mask = (target_kelvin >= low) & (target_kelvin <= high)
        else:
            mask = (target_kelvin >= low) & (target_kelvin < high)
        result = torch.where(mask, per_range_var[i], result)
    return result


# ============================================================
# METRICS
# ============================================================

def calculate_basic_metrics(prediction, target):
    error = prediction - target
    mse = (error ** 2).mean().item()
    rmse = math.sqrt(mse)
    mae = error.abs().mean().item()
    bias = error.mean().item()
    return {"mse": mse, "rmse": rmse, "mae": mae, "bias": bias}


def calculate_range_metrics(prediction, target, ranges):
    results = []
    for i, (low, high) in enumerate(ranges):
        if i == len(ranges) - 1:
            mask = (target >= low) & (target <= high)
        else:
            mask = (target >= low) & (target < high)

        count = int(mask.sum().item())
        if count == 0:
            results.append({"range": f"{low:.0f}-{high:.0f}", "nodes": 0,
                             "rmse": None, "mae": None, "bias": None})
            continue

        metrics = calculate_basic_metrics(prediction[mask], target[mask])
        results.append({"range": f"{low:.0f}-{high:.0f}", "nodes": count,
                         "rmse": metrics["rmse"], "mae": metrics["mae"], "bias": metrics["bias"]})
    return results


def calculate_fine_temperature_diagnostics(prediction, target):
    diagnostics = []
    for i, (low, high) in enumerate(DIAGNOSTIC_BINS):
        if i == len(DIAGNOSTIC_BINS) - 1:
            mask = (target >= low) & (target <= high)
        else:
            mask = (target >= low) & (target < high)

        count = int(mask.sum().item())
        if count == 0:
            diagnostics.append({"range": f"{low:.0f}-{high:.0f}", "nodes": 0, "rmse": None, "mae": None})
            continue

        error = prediction[mask] - target[mask]
        diagnostics.append({
            "range": f"{low:.0f}-{high:.0f}",
            "nodes": count,
            "rmse": math.sqrt((error ** 2).mean().item()),
            "mae": error.abs().mean().item(),
        })
    return diagnostics


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, ml_criterion, real_criterion, optimizer, physics_lambda):
    model.train()

    total_loss_sum = 0.0
    ml_loss_sum = 0.0
    real_loss_sum = 0.0
    physics_loss_sum = 0.0
    num_batches = 0

    all_predictions = []
    all_targets = []

    global_residual_abs = 0.0
    global_residual_count = 0

    normalized_sq_error = 0.0
    normalized_abs_error = 0.0
    normalized_count = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        prediction_normalized = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.global_features[batch.batch],
        )

        target_normalized = batch.T_real_normalized
        target_kelvin = batch.y

        if prediction_normalized.shape != target_normalized.shape:
            raise RuntimeError(
                f"Prediction shape {prediction_normalized.shape} != target shape {target_normalized.shape}"
            )

        prediction_kelvin = normalized_to_kelvin(prediction_normalized)

        ml_loss = ml_criterion(prediction_normalized, target_normalized, target_kelvin)
        real_loss = real_criterion(prediction_kelvin, target_kelvin)
        physics_loss, global_residual = calculate_global_energy_loss(prediction_kelvin, batch)

        total_loss = (
            ml_loss
            + CONFIG["real_space_loss_weight"] * real_loss
            + physics_lambda * physics_loss
        )

        total_loss.backward()

        # for check the gradients...
        #grad_norm = torch.sqrt(
        #sum(
        #    p.grad.detach().norm(2) ** 2
        #    for p in model.parameters()
        #    if p.grad is not None))
        #
        #print(f"Gradient norm: {grad_norm.item():.4f}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()

        with torch.no_grad():
            normalized_error = prediction_normalized - target_normalized
            normalized_sq_error += (normalized_error ** 2).sum().item()
            normalized_abs_error += normalized_error.abs().sum().item()
            normalized_count += target_normalized.numel()

            all_predictions.append(prediction_kelvin.detach().cpu())
            all_targets.append(target_kelvin.detach().cpu())

            eligible = batch.pinn_global_energy_eligible.reshape(-1).bool()
            if eligible.any():
                global_residual_abs += global_residual[eligible].abs().sum().item()
                global_residual_count += int(eligible.sum().item())

        total_loss_sum += total_loss.item()
        ml_loss_sum += ml_loss.item()
        real_loss_sum += real_loss.item()
        physics_loss_sum += physics_loss.item()
        num_batches += 1

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    normalized_rmse = math.sqrt(normalized_sq_error / max(normalized_count, 1))
    normalized_mae = normalized_abs_error / max(normalized_count, 1)

    real_metrics = calculate_basic_metrics(all_predictions, all_targets)
    global_residual_mae = global_residual_abs / max(global_residual_count, 1)

    return {
        "total_loss": total_loss_sum / max(num_batches, 1),
        "ml_loss": ml_loss_sum / max(num_batches, 1),
        "real_loss": real_loss_sum / max(num_batches, 1),
        "physics_loss": physics_loss_sum / max(num_batches, 1),
        "normalized_rmse": normalized_rmse,
        "normalized_mae": normalized_mae,
        "real_rmse": real_metrics["rmse"],
        "real_mae": real_metrics["mae"],
        "real_bias": real_metrics["bias"],
        "global_residual_mae_W": global_residual_mae,
        "predictions": all_predictions,
        "targets": all_targets,
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader, per_range_var):
    model.eval()

    all_predictions = []
    all_targets = []

    normalized_sq_error = 0.0
    normalized_abs_error = 0.0
    normalized_count = 0

    physics_loss_sum = 0.0
    physics_batches = 0

    global_residual_abs = 0.0
    global_residual_count = 0

    for batch in loader:
        batch = batch.to(device)

        prediction_normalized = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.global_features[batch.batch],
        )

        target_normalized = batch.T_real_normalized
        target_kelvin = batch.y

        normalized_error = prediction_normalized - target_normalized
        normalized_sq_error += (normalized_error ** 2).sum().item()
        normalized_abs_error += normalized_error.abs().sum().item()
        normalized_count += target_normalized.numel()

        # Per-node Jensen correction using the variance of the range
        # each node's TARGET temperature falls in. (Target is used, not
        # prediction, purely to pick which range's noise level applies;
        # this is a diagnostic/correction lookup, not a leak into the
        # training loss -- validate() never backprops.)
        node_var = per_node_range_variance(target_kelvin, per_range_var)
        prediction_kelvin = apply_jensen_correction(prediction_normalized, node_var)

        all_predictions.append(prediction_kelvin.cpu())
        all_targets.append(target_kelvin.cpu())

        raw_prediction_kelvin = normalized_to_kelvin(prediction_normalized)
        val_physics_loss, global_residual = calculate_global_energy_loss(raw_prediction_kelvin, batch)

        physics_loss_sum += val_physics_loss.item()
        physics_batches += 1

        eligible = batch.pinn_global_energy_eligible.reshape(-1).bool()
        if eligible.any():
            global_residual_abs += global_residual[eligible].abs().sum().item()
            global_residual_count += int(eligible.sum().item())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    normalized_rmse = math.sqrt(normalized_sq_error / max(normalized_count, 1))
    normalized_mae = normalized_abs_error / max(normalized_count, 1)

    real_metrics = calculate_basic_metrics(all_predictions, all_targets)
    physics_loss = physics_loss_sum / max(physics_batches, 1)
    global_residual_mae = global_residual_abs / max(global_residual_count, 1)

    return {
        "normalized_rmse": normalized_rmse,
        "normalized_mae": normalized_mae,
        "real_rmse": real_metrics["rmse"],
        "real_mae": real_metrics["mae"],
        "real_bias": real_metrics["bias"],
        "physics_loss": physics_loss,
        "global_residual_mae_W": global_residual_mae,
        "predictions": all_predictions,
        "targets": all_targets,
    }


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(model, train_loader, val_loader):
    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------
    assert_all_nodes_are_targets(train_loader)
    assert_physics_fields_exist(train_loader)
    compute_temperature_distribution(train_loader)

    assert_all_nodes_are_targets(val_loader)
    assert_physics_fields_exist(val_loader)

    ml_criterion = RangeBalancedMLLoss(
        TEMPERATURE_BINS,
        BOUNDARY_MARGIN_K
    ).to(device)

    real_criterion = RangeBalancedRealSpaceLoss(
        TEMPERATURE_BINS,
        BOUNDARY_MARGIN_K,
        scale_K=CONFIG["real_space_loss_scale_K"]
    ).to(device)

    # From-scratch optimizer.
    # No checkpoint/previous optimizer state is loaded.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=CONFIG["scheduler_factor"],
        patience=CONFIG["scheduler_patience"]
    )

    best_RMSE = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print("\n" + "=" * 80)
    print("ULTRA SUPER POWER MODEL")
    print("FROM-SCRATCH ML + PHYSICS TRAINING")
    print("=" * 80)
    print("\nTraining dataset:")
    print("  Original Train + 450-500 K + 400-450 K + Hard Graphs")
    print("\nValidation dataset:")
    print("  ORIGINAL validation loader only (loaders['val'])")
    print("\nTest dataset:")
    print("  ORIGINAL test loader only (not used during training)")
    print("\nTemperature objective:")
    for low, high in TEMPERATURE_BINS:
        print(f"  {low:.0f}-{high:.0f} K -> equal contribution")

    print(f"\nBoundary soft-transition margin: {BOUNDARY_MARGIN_K:.1f} K")
    print(f"Real-space loss weight: {CONFIG['real_space_loss_weight']}")
    print(f"Physics lambda: {CONFIG['lambda_physics']}")
    print(f"Device: {device}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print("=" * 80)

    for epoch in range(CONFIG["epochs"]):

        # --------------------------------------------------------
        # Physics warmup
        # --------------------------------------------------------
        warmup_epochs = CONFIG["physics_warmup_epochs"]

        if epoch + 1 <= warmup_epochs:
            physics_lambda = CONFIG["lambda_physics"] * (
                (epoch + 1) / warmup_epochs
            )
        else:
            physics_lambda = CONFIG["lambda_physics"]

        # --------------------------------------------------------
        # TRAIN ONLY ON Fine_tuning_data
        # --------------------------------------------------------
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            ml_criterion=ml_criterion,
            real_criterion=real_criterion,
            optimizer=optimizer,
            physics_lambda=physics_lambda,
        )

        # Jensen correction is estimated ONLY from the training data.
        # Validation remains untouched and is never used for fitting.
        per_range_var = estimate_jensen_sigma_per_range(
            model,
            train_loader
        )

        # --------------------------------------------------------
        # VALIDATION ONLY ON ORIGINAL loaders["val"]
        # --------------------------------------------------------
        val_metrics = validate(
            model=model,
            loader=val_loader,
            per_range_var=per_range_var
        )

        gap = 100.0 * (
            (val_metrics["real_rmse"] - train_metrics["real_rmse"])
            / max(train_metrics["real_rmse"], 1e-12)
        )

        train_range_metrics = calculate_range_metrics(
            train_metrics["predictions"],
            train_metrics["targets"],
            TEMPERATURE_BINS
        )

        val_range_metrics = calculate_range_metrics(
            val_metrics["predictions"],
            val_metrics["targets"],
            TEMPERATURE_BINS
        )

        train_fine_bins = calculate_fine_temperature_diagnostics(
            train_metrics["predictions"],
            train_metrics["targets"]
        )

        val_fine_bins = calculate_fine_temperature_diagnostics(
            val_metrics["predictions"],
            val_metrics["targets"]
        )

        print(f"\nEpoch {epoch + 1:03d}")
        print(f"Physics lambda: {physics_lambda:.5f}")

        print(
            f"Train | Total: {train_metrics['total_loss']:.6f} "
            f"| ML: {train_metrics['ml_loss']:.6f} "
            f"| Real: {train_metrics['real_loss']:.6f} "
            f"| Physics: {train_metrics['physics_loss']:.8f}"
        )

        print(
            f"Train | Real RMSE: {train_metrics['real_rmse']:.4f} K "
            f"| Real MAE: {train_metrics['real_mae']:.4f} K "
            f"| Real Bias: {train_metrics['real_bias']:+.4f} K"
        )

        print(
            f"Validation | Real RMSE: {val_metrics['real_rmse']:.4f} K "
            f"| Real MAE: {val_metrics['real_mae']:.4f} K "
            f"| Real Bias: {val_metrics['real_bias']:+.4f} K"
        )

        print(f"Train/Val gap: {gap:+.2f}%")

        print(
            "Per-range Jensen sigma: "
            + ", ".join(
                f"[{low:.0f}-{high:.0f}]={math.sqrt(per_range_var[i].item()):.4f}"
                for i, (low, high) in enumerate(TEMPERATURE_BINS)
            )
        )

        print(f"LR: {optimizer.param_groups[0]['lr']:.8f}")

        print("\nTemperature-range validation:")
        print("-" * 70)

        for item in val_range_metrics:
            if item["nodes"] == 0:
                print(f"{item['range']} K | No nodes")
            else:
                print(
                    f"{item['range']} K | Nodes: {item['nodes']:7d} | "
                    f"RMSE: {item['rmse']:8.4f} K | "
                    f"MAE: {item['mae']:8.4f} K | "
                    f"Bias: {item['bias']:+8.4f} K"
                )

        # --------------------------------------------------------
        # SAVE BEST MODEL BEFORE THE GAP GUARD
        #
        # This is important: if an epoch achieves the best validation
        # RMSE but has an unsafe train/val gap, we still retain that
        # exact model as the best candidate instead of losing it.
        # --------------------------------------------------------
        is_best = val_metrics["real_rmse"] < best_RMSE

        if is_best:
            best_RMSE = val_metrics["real_rmse"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": CONFIG,
                    "epoch": epoch + 1,
                    "rmse": val_metrics["real_rmse"],
                    "mae": val_metrics["real_mae"],
                    "bias": val_metrics["real_bias"],
                    "per_range_jensen_var": per_range_var.detach().cpu().tolist(),
                    "targets": "all_nodes",
                    "physics": "global_energy_only",
                    "temperature_balance": "soft_equal_contribution_per_range",
                    "temperature_ranges": TEMPERATURE_BINS,
                    "boundary_margin_K": BOUNDARY_MARGIN_K,
                    "physics_lambda": physics_lambda,
                    "physics_loss": val_metrics["physics_loss"],
                    "global_residual_mae_W": val_metrics["global_residual_mae_W"],
                    "training_mode": "from_scratch",
                    "training_dataset": (
                        "original_train + 450_500_high_temp + "
                        "400_450_high_temp + hard_graphs"
                    ),
                    "validation_dataset": "original_loaders_val",
                    "test_used_during_training": False,
                },
                CONFIG["checkpoint_path"],
            )

            print(
                f"\n>>> BEST ULTRA SUPER POWER MODEL SAVED <<< "
                f"(Validation Real RMSE: {best_RMSE:.4f} K)"
            )
        else:
            epochs_without_improvement += 1

        # --------------------------------------------------------
        # SAVE HISTORY
        # --------------------------------------------------------
        history.append({
            "epoch": epoch + 1,
            "train_total_loss": train_metrics["total_loss"],
            "train_ml_loss": train_metrics["ml_loss"],
            "train_real_loss": train_metrics["real_loss"],
            "train_physics_loss": train_metrics["physics_loss"],
            "train_real_rmse": train_metrics["real_rmse"],
            "train_real_mae": train_metrics["real_mae"],
            "train_real_bias": train_metrics["real_bias"],
            "val_real_rmse": val_metrics["real_rmse"],
            "val_real_mae": val_metrics["real_mae"],
            "val_real_bias": val_metrics["real_bias"],
            "val_physics_loss": val_metrics["physics_loss"],
            "train_val_gap_pct": gap,
            "per_range_jensen_var": per_range_var.detach().cpu().tolist(),
            "physics_lambda": physics_lambda,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_range_metrics": train_range_metrics,
            "val_range_metrics": val_range_metrics,
            "train_fine_10K_bins": train_fine_bins,
            "val_fine_10K_bins": val_fine_bins,
            "is_best": is_best,
        })

        results = {
            "model_name": "Ultra Super Power Model",
            "best_validation_real_rmse": best_RMSE,
            "best_epoch": best_epoch,
            "temperature_balance": "soft_equal_contribution_per_range",
            "boundary_margin_K": BOUNDARY_MARGIN_K,
            "temperature_ranges": TEMPERATURE_BINS,
            "physics": "global_energy_only",
            "targets": "all_nodes",
            "training_mode": "from_scratch",
            "training_dataset": {
                "original_train": len(original_train_dataset),
                #"above450": len(above450_dataset_processed),
                #"above400_450": len(above400_dataset_processed),
                #"hard_graphs": len(hard_graphs_dataset_processed),
                "total": len(Fine_tuning_data),
            },
            "validation_dataset": "loaders['val']",
            "test_used_during_training": False,
            "history": history,
        }

        with open(CONFIG["results_path"], "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Scheduler sees ONLY validation RMSE.
        scheduler.step(val_metrics["real_rmse"])

        # --------------------------------------------------------
        # GAP-BASED OVERFITTING GUARD
        # --------------------------------------------------------
        past_warmup = (epoch + 1) > (
            CONFIG["physics_warmup_epochs"] + CONFIG["gap_guard_buffer_epochs"]
        )

        gap_unsafe = (
            gap > CONFIG["max_safe_train_val_gap_pct"]
        )

        if past_warmup and gap_unsafe:
            print(
                f"\n⚠️ STOPPING: train/val gap ({gap:+.2f}%) "
                f"exceeded the safe threshold "
                f"({CONFIG['max_safe_train_val_gap_pct']:.1f}%)."
            )
            print(
                f"Best checkpoint: epoch {best_epoch} "
                f"| Validation Real RMSE: {best_RMSE:.4f} K"
            )
            break

        if epochs_without_improvement >= CONFIG["early_stopping_patience"]:
            print("\nEarly stopping triggered.")
            print(
                f"No validation improvement for "
                f"{epochs_without_improvement} epochs."
            )
            break

        print("*" * 80)

    print("\n" + "=" * 80)
    print("ULTRA SUPER POWER TRAINING FINISHED")
    print("=" * 80)
    print(f"Best validation Real RMSE: {best_RMSE:.4f} K")
    print(f"Best epoch: {best_epoch}")
    print(f"Checkpoint: {CONFIG['checkpoint_path']}")
    print(f"Results: {CONFIG['results_path']}")
    print(f"Device used: {device}")
    print("=" * 80)

    return model, history


# ============================================================
# CREATE MODEL AND TRAIN
# ============================================================

model = ML_only_model(
    in_features=CONFIG["in_features"],
    hidden_features=CONFIG["hidden_features"],
    in_features_edge=CONFIG["in_features_edge"],
    global_features=CONFIG["global_features"],
    num_layers=CONFIG["num_layers"],
    edge_mlp_hidden=CONFIG["edge_mlp_hidden"],
    gat_heads=CONFIG["gat_heads"],
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n" + "=" * 40)
print("MODEL")
print("=" * 40)
print(model)
print(f"\nTrainable parameters: {num_params:,}")
print(f"Message-passing layers: {CONFIG['num_layers']}")
print(f"Hidden features: {CONFIG['hidden_features']}")

model, history = train_model(
    model=model,
    train_loader=Fine_tuning_loader,
    val_loader=validation_loader,
)