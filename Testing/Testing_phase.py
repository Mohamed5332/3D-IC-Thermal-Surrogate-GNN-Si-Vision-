
print("Testing in action...")


import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


import torch
from joblib import load
import torch.nn as nn 
import torch_geometric.nn as geometric_nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Plotting_results.plot_predictionsVSGroundTruth import plot_graph


CONFIG = {
    # ---- architecture ----
    "in_features":       18,
    "in_features_edge":  11,
    "global_features":   21,
    "hidden_features":   64, # i edited here...
    "edge_mlp_hidden":   32, 
    "num_layers":        4,
    "gat_heads":         2,
#ML_with_physics_model_range_balanced.pt
#ML_with_physics_model_hard_finetuned.pt
#ML_with_physics_model_hard_finetuned_x3.pt
#ML_with_physics_model_hard_finetuned_super_powr.pt
#Model_v2_playground.pt
#Model_v1_playground.pt
#ML_with_physics_model_ultra_super_power_V2000.pt
    # ---- checkpoint ----
    "checkpoint_path":   r"C:/msys64/home/dell/Model_v7_playground.pt",

    # ---- Jensen correction ----
    "use_lognormal_correction": True,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ML_only_model(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features,
        in_features_edge,
        global_features,
        num_layers=4,
        edge_mlp_hidden=32,
        gat_heads=2,
    ):

        super().__init__()

        # ====================================================
        # NODE INPUT PROJECTION
        # ====================================================

        self.input_proj = nn.Sequential(

            nn.Linear(
                in_features,
                hidden_features
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_features,
                hidden_features
            ),
        )

        # ====================================================
        # MESSAGE PASSING
        # ====================================================

        self.nn_convs = nn.ModuleList()
        self.gat_convs = nn.ModuleList()
        self.merges = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):

            # -------------------------------
            # Edge MLP
            # -------------------------------

            edge_mlp = nn.Sequential(

                nn.Linear(
                    in_features_edge,
                    edge_mlp_hidden
                ),

                nn.ReLU(),

                nn.Linear(
                    edge_mlp_hidden,
                    hidden_features * hidden_features
                ),
            )

            # -------------------------------
            # NNConv
            # -------------------------------

            self.nn_convs.append(

                geometric_nn.NNConv(

                    hidden_features,
                    hidden_features,

                    edge_mlp,

                    aggr="mean",
                )
            )

            # -------------------------------
            # GATv2
            # -------------------------------

            self.gat_convs.append(

                geometric_nn.GATv2Conv(

                    in_channels=hidden_features,

                    out_channels=hidden_features,

                    heads=gat_heads,

                    concat=False,

                    edge_dim=in_features_edge,

                    add_self_loops=False,
                )
            )

            # -------------------------------
            # Merge
            # -------------------------------

            self.merges.append(

                nn.Linear(
                    hidden_features * 2,
                    hidden_features
                )
            )

            # -------------------------------
            # LayerNorm
            # -------------------------------

            self.norms.append(

                nn.LayerNorm(
                    hidden_features
                )
            )

        # ====================================================
        # GLOBAL ENCODER
        # ====================================================

        self.global_encoder = nn.Sequential(

            nn.Linear(
                global_features,
                hidden_features
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_features,
                hidden_features
            ),
        )

        # ====================================================
        # FINAL PREDICTION HEAD
        # ====================================================

        self.FC = nn.Sequential(

            nn.Linear(
                hidden_features * 2,
                hidden_features
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_features,
                1
            ),
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        global_features
    ):

        # ====================================================
        # NODE PROJECTION
        # ====================================================

        h = self.input_proj(x)

        # ====================================================
        # MESSAGE PASSING
        # ====================================================

        for conv_nn, conv_gat, merge, norm in zip(

            self.nn_convs,
            self.gat_convs,
            self.merges,
            self.norms
        ):

            residual = h

            # NNConv
            h_nn = conv_nn(
                h,
                edge_index,
                edge_attr
            )

            # GATv2
            h_gat = conv_gat(
                h,
                edge_index,
                edge_attr
            )

            # Merge
            h = merge(
                torch.cat(
                    [
                        h_nn,
                        h_gat
                    ],
                    dim=-1
                )
            )

            h = F.relu(h)

            # Residual
            h = h + residual

            # LayerNorm
            h = norm(h)

        # ====================================================
        # GLOBAL ENCODER
        # ====================================================

        global_embedding = self.global_encoder(
            global_features
        )

        # ====================================================
        # NODE + GLOBAL
        # ====================================================

        final_features = torch.cat(
            [
                h,
                global_embedding
            ],
            dim=-1
        )

        # ====================================================
        # PREDICTION
        # ====================================================

        return self.FC(
            final_features
        ).squeeze(-1)


checkpoint = torch.load(
    CONFIG["checkpoint_path"],
    map_location=device,
    weights_only=False
)


loaders = load(filename="loaders")

model = ML_only_model(in_features=CONFIG['in_features'] , 
                      hidden_features= CONFIG['hidden_features'] , 
                      in_features_edge= CONFIG['in_features_edge'] , 
                      global_features=CONFIG['global_features'] , 
                      num_layers=CONFIG['num_layers'] , 
                      edge_mlp_hidden=CONFIG['edge_mlp_hidden'] , 
                      gat_heads=2).to(device)

model.load_state_dict(state_dict= checkpoint['model_state_dict'])


normalized_temp_scaler = load(
    filename="normalized_temp_scaller"
)

std_target_temp = normalized_temp_scaler["std_target_temp"]

target_temp_mean = torch.tensor(
    std_target_temp.mean_[0],
    dtype=torch.float32,
    device=device
)

target_temp_scale = torch.tensor(
    std_target_temp.scale_[0],
    dtype=torch.float32,
    device=device
)


print("Model : " , model)


def test_full_dataset(model, test_loader, target_temp_mean, target_temp_scale, device):
    """
    Run inference on ALL graphs in the test loader.

    Returns:
        results: dictionary containing overall and per-range metrics
        all_predictions: list of prediction tensors, one per graph
        all_ground_truth: list of ground-truth tensors, one per graph
    """

    model.eval()

    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():

        for batch in test_loader:

            batch = batch.to(device)

            # ------------------------------------------------
            # Node features
            # ------------------------------------------------

            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr

            # ------------------------------------------------
            # Global features
            #
            # One global vector per graph.
            # batch.batch tells us which graph each node belongs to.
            # ------------------------------------------------

            global_features = batch.global_features[batch.batch]

            # ------------------------------------------------
            # Model prediction
            # ------------------------------------------------

            output = model(
                x,
                edge_index,
                edge_attr,
                global_features
            )

            # ------------------------------------------------
            # Convert normalized/log prediction back to Kelvin
            # ------------------------------------------------

            prediction = torch.exp(
                output * target_temp_scale + target_temp_mean
            )

            ground_truth = batch.y.view(-1)

            # ------------------------------------------------
            # Split predictions back into individual graphs
            # ------------------------------------------------

            num_graphs = batch.num_graphs

            for graph_idx in range(num_graphs):

                node_mask = (batch.batch == graph_idx)

                graph_prediction = prediction[node_mask].detach().cpu()
                graph_ground_truth = ground_truth[node_mask].detach().cpu()

                all_predictions.append(graph_prediction)
                all_ground_truth.append(graph_ground_truth)

    # ========================================================
    # Concatenate ALL test nodes
    # ========================================================

    predictions = torch.cat(all_predictions)
    ground_truth = torch.cat(all_ground_truth)

    errors = predictions - ground_truth
    abs_errors = torch.abs(errors)

    # ========================================================
    # Overall metrics
    # ========================================================

    overall_rmse = torch.sqrt(
        torch.mean(errors ** 2)
    ).item()

    overall_mae = torch.mean(
        abs_errors
    ).item()

    overall_bias = torch.mean(
        errors
    ).item()

    max_error = torch.max(
        abs_errors
    ).item()

    # ========================================================
    # Error percentages
    # ========================================================

    total_nodes = len(errors)

    error_over_5 = (
        (abs_errors > 5).sum().item()
        / total_nodes
        * 100
    )

    error_over_8 = (
        (abs_errors > 8).sum().item()
        / total_nodes
        * 100
    )

    error_over_10 = (
        (abs_errors > 10).sum().item()
        / total_nodes
        * 100
    )

    # ========================================================
    # Per temperature range
    # ========================================================

    temperature_ranges = {
        "300-350 K": (300.0, 350.0),
        "350-400 K": (350.0, 400.0),
        "400-450 K": (400.0, 450.0),
        "450-500 K": (450.0, 500.0),
    }

    per_range = {}

    for range_name, (low, high) in temperature_ranges.items():

        mask = (
            (ground_truth >= low)
            & (ground_truth < high)
        )

        if mask.sum() == 0:
            continue

        range_errors = errors[mask]
        range_abs_errors = abs_errors[mask]

        range_rmse = torch.sqrt(
            torch.mean(range_errors ** 2)
        ).item()

        range_mae = torch.mean(
            range_abs_errors
        ).item()

        range_bias = torch.mean(
            range_errors
        ).item()

        range_max_error = torch.max(
            range_abs_errors
        ).item()

        range_nodes = mask.sum().item()

        range_over_8 = (
            (range_abs_errors > 8).sum().item()
            / range_nodes
            * 100
        )

        range_over_10 = (
            (range_abs_errors > 10).sum().item()
            / range_nodes
            * 100
        )

        per_range[range_name] = {
            "nodes": range_nodes,
            "rmse": range_rmse,
            "mae": range_mae,
            "bias": range_bias,
            "max_error": range_max_error,
            "error_over_8_pct": range_over_8,
            "error_over_10_pct": range_over_10,
        }

    # ========================================================
    # Print results
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FULL TEST SET RESULTS")
    print("=" * 70)

    print(f"Number of test graphs : {len(all_predictions)}")
    print(f"Total test nodes      : {total_nodes}")

    print("\nOverall:")
    print("-" * 70)
    print(f"RMSE       : {overall_rmse:.4f} K")
    print(f"MAE        : {overall_mae:.4f} K")
    print(f"Bias       : {overall_bias:+.4f} K")
    print(f"Max Error  : {max_error:.4f} K")

    print("\nError distribution:")
    print("-" * 70)
    print(f"Error > 5 K   : {error_over_5:.2f}%")
    print(f"Error > 8 K   : {error_over_8:.2f}%")
    print(f"Error > 10 K  : {error_over_10:.2f}%")

    print("\nTemperature-range test:")
    print("-" * 70)

    for range_name, metrics in per_range.items():

        print(
            f"{range_name:12s} | "
            f"Nodes: {metrics['nodes']:6d} | "
            f"RMSE: {metrics['rmse']:8.4f} K | "
            f"MAE: {metrics['mae']:8.4f} K | "
            f"Bias: {metrics['bias']:+8.4f} K | "
            f">8K: {metrics['error_over_8_pct']:6.2f}% | "
            f">10K: {metrics['error_over_10_pct']:6.2f}%"
        )

    print("=" * 70)

    # ========================================================
    # Return everything
    # ========================================================

    results = {
        "num_graphs": len(all_predictions),
        "total_nodes": total_nodes,

        "overall": {
            "rmse": overall_rmse,
            "mae": overall_mae,
            "bias": overall_bias,
            "max_error": max_error,
            "error_over_5_pct": error_over_5,
            "error_over_8_pct": error_over_8,
            "error_over_10_pct": error_over_10,
        },

        "per_range": per_range,
    }

    return results, all_predictions, all_ground_truth




results, all_predictions, all_ground_truth = test_full_dataset(
    model=model,
    test_loader=loaders["test"],
    target_temp_mean=target_temp_mean,
    target_temp_scale=target_temp_scale,
    device=device
)