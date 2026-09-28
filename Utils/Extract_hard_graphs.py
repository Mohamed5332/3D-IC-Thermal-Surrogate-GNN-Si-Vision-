"""
============================================================
EXTRACT HARD/PROBLEM GRAPHS FOR MANUAL OVERSAMPLING
============================================================

الهدف:
تدخل على الداتاسيت الكامل بتاعك (كل الجرافات، مش بس test)،
تعمل نفس الـ preprocessing المستخدم أثناء التدريب،
تشغّل الموديل المدرب عليهم، وتطلع subset فيه بس الجرافات اللي:

    (أ) عندها |bias| كبير فعليًا

    أو

    (ب) موجودة ضمن أسوأ نسبة من الجرافات حسب |bias|

مهم:
الـ preprocessing يتم على نسخة من الجرافات فقط،
والـ raw graphs الأصلية تظل بدون تعديل.

الناتج:
ملف .pt جديد فيه list من الـ RAW Data objects فقط،
تقدر تضيفه/تكرره جوه الـ training set بتاعك
(oversampling).

============================================================
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geometric_nn

from torch_geometric.loader import DataLoader
from joblib import load

import numpy as np
import pandas as pd


# ============================================================
# PATHS - EDIT THESE ONLY
# ============================================================

DATASET_PATH = r"D:/Thermal surrogate project/train_raw_graphs.pt"

CHECKPOINT_PATH = r"ML_with_physics_model_range_balanced.pt"

OUTPUT_PATH = r"D:/Thermal surrogate project/hard_graphs_subset.pt"


# ============================================================
# PREPROCESSING FILES
# ============================================================

NODE_SCALERS_PATH = r"scallers_Node_data"

EDGE_SCALERS_PATH = r"scallers_Edege_data"

GLOBAL_SCALERS_PATH = r"scallers_global_data"

NODE_ONE_HOT_PATH = r"one_hot_Node_data"

EDGE_ONE_HOT_PATH = r"one_hot_edge_data"

GLOBAL_ONE_HOT_PATH = r"one_hot_global_data"

TEMP_SCALER_PATH = r"normalized_temp_scaller"


# ============================================================
# EXTRACTION SETTINGS
# ============================================================

# أي graph عنده |bias| أكبر أو يساوي القيمة دي يتم اختياره
ABS_BIAS_THRESHOLD_K = 8.0

# بالإضافة إلى ذلك، نأخذ أسوأ 10% من الجرافات حسب |bias|
TOP_PERCENTILE = 0.10

# Batch size أثناء inference
BATCH_SIZE = 32


# ============================================================
# MODEL CONFIG
# ============================================================

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

    "boundary_margin_K": 10.0,

    "real_space_loss_weight": 0.5,

    "real_space_loss_scale_K": 10.0,

    "lambda_physics": 0.05,

    "physics_warmup_epochs": 15,

    "lr": 1e-3,

    "weight_decay": 3e-4,

    "grad_clip": 5.0,

    "epochs": 200,

    "scheduler_factor": 0.5,

    "scheduler_patience": 2,

    "early_stopping_patience": 12,

    "max_safe_train_val_gap_pct": 15.0,

    "use_lognormal_correction": True,

    "diagnostic_bin_size_K": 10.0,

    "checkpoint_path":
        "ML_with_physics_model_range_balanced.pt",

    "results_path":
        "ML_with_physics_training_results_range_balanced.json",
}


# ============================================================
# LOAD PREPROCESSING OBJECTS
# ============================================================

def load_preprocessing_objects():

    print("\n")
    print("=" * 60)
    print("LOADING PREPROCESSING OBJECTS")
    print("=" * 60)

    # --------------------------------------------------------
    # Scalers
    # --------------------------------------------------------

    print("\nLoading node scalers...")

    Train_Nodes_scallers = load(
        filename=NODE_SCALERS_PATH
    )

    print("Loading edge scalers...")

    Train_Edges_scallers = load(
        filename=EDGE_SCALERS_PATH
    )

    print("Loading global scalers...")

    Train_global_scallers = load(
        filename=GLOBAL_SCALERS_PATH
    )

    # --------------------------------------------------------
    # One-hot metadata
    # --------------------------------------------------------

    print("Loading node one-hot metadata...")

    Train_one_hot_Nodes_data = load(
        filename=NODE_ONE_HOT_PATH
    )

    print("Loading edge one-hot metadata...")

    Train_one_hot_Edges_data = load(
        filename=EDGE_ONE_HOT_PATH
    )

    print("Loading global one-hot metadata...")

    Train_one_hot_global_data = load(
        filename=GLOBAL_ONE_HOT_PATH
    )

    # --------------------------------------------------------
    # Target scaler
    # --------------------------------------------------------

    print("Loading target temperature scaler...")

    normalized_temp_scaller = load(
        filename=TEMP_SCALER_PATH
    )

    print("\nPreprocessing objects loaded successfully.")

    return (
        Train_Nodes_scallers,
        Train_Edges_scallers,
        Train_global_scallers,
        Train_one_hot_Nodes_data,
        Train_one_hot_Edges_data,
        Train_one_hot_global_data,
        normalized_temp_scaller,
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_graph_list(
    graph_list,
    Train_Nodes_scallers,
    Train_Edges_scallers,
    Train_global_scallers,
    Train_one_hot_Nodes_data,
    Train_one_hot_Edges_data,
    Train_one_hot_global_data,
):
    """
    Apply the EXACT preprocessing pipeline used during training.

    IMPORTANT:
    The original graph_list is NOT modified.

    A deep copy is created and preprocessing is applied
    to that copy.

    The returned graphs are the graphs used for inference.
    """

    print("\n")
    print("=" * 60)
    print("PREPROCESSING GRAPHS")
    print("=" * 60)

    # ========================================================
    # Load metadata
    # ========================================================

    Number_of_layers = (
        Train_one_hot_Nodes_data[
            "Number_of_layers"
        ]
    )

    Number_of_block_types = (
        Train_one_hot_Nodes_data[
            "Number_of_block_types"
        ]
    )

    Number_of_edge_types = (
        Train_one_hot_Edges_data[
            "Number_of_edge_types"
        ]
    )

    Number_of_cooling_types = (
        Train_one_hot_global_data[
            "Number_of_cooling_types"
        ]
    )

    Number_of_package_recipe_code = (
        Train_one_hot_global_data[
            "Number_of_package_recipe_code"
        ]
    )

    # ========================================================
    # Node scalers
    # ========================================================

    std_power_density = (
        Train_Nodes_scallers[
            "std_power_density"
        ]
    )

    std_area = (
        Train_Nodes_scallers[
            "std_area"
        ]
    )

    std_kappa = (
        Train_Nodes_scallers[
            "std_kappa"
        ]
    )

    std_exposed_area_mm2 = (
        Train_Nodes_scallers[
            "std_exposed_area_mm2"
        ]
    )

    # ========================================================
    # Edge scalers
    # ========================================================

    std_resistance = (
        Train_Edges_scallers[
            "std_resistance"
        ]
    )

    std_geometric_distance = (
        Train_Edges_scallers[
            "std_geometric_distance"
        ]
    )

    std_contact_area = (
        Train_Edges_scallers[
            "std_contact_area"
        ]
    )

    std_thermal_path_length = (
        Train_Edges_scallers[
            "std_thermal_path_length"
        ]
    )

    # ========================================================
    # Global scalers
    # ========================================================

    std_total_power_W = (
        Train_global_scallers[
            "std_total_power_W"
        ]
    )

    std_cooling_htc = (
    Train_global_scallers[
            "std_cooling_htc"
        ]
    )

    std_bond_kappa_w_mk = (
        Train_global_scallers[
            "std_bond_kappa_w_mk"
        ]
    )

    std_bond_thickness_mm = (
        Train_global_scallers[
            "std_bond_thickness_mm"
        ]
    )

    std_ambient_temp_K = (
        Train_global_scallers[
            "std_ambient_temp_K"
        ]
    )

    std_workload_factor = (
        Train_global_scallers[
            "std_workload_factor"
        ]
    )

    std_n_tsv_pairs = (
        Train_global_scallers[
            "std_n_tsv_pairs"
        ]
    )

    std_hotspot_multiplier = (
        Train_global_scallers[
            "std_hotspot_multiplier"
        ]
    )

    std_chip_width_mm = (
        Train_global_scallers[
            "std_chip_width_mm"
        ]
    )

    std_chip_height_mm = (
        Train_global_scallers[
            "std_chip_height_mm"
        ]
    )

    # ========================================================
    # Deep copy
    # ========================================================

    print(
        "\nCreating independent copy of graphs..."
    )

    processed_graphs = copy.deepcopy(
        graph_list
    )

    # ========================================================
    # Process each graph
    # ========================================================

    total_graphs = len(processed_graphs)

    for graph_index, graph in enumerate(
        processed_graphs
    ):

        # ====================================================
        # NODE PREPROCESSING
        # ====================================================

        keep = [
            0,
            1,
            3,
            5,
            6,
            12
        ]

        # ----------------------------------------------------
        # One-hot layer index
        # ----------------------------------------------------

        encoded_layer_idx = F.one_hot(
            graph.x[:, 2].long(),
            Number_of_layers
        )

        # ----------------------------------------------------
        # One-hot block type
        # ----------------------------------------------------

        encoded_block_type = F.one_hot(
            graph.x[:, 4].long(),
            Number_of_block_types
        )

        # ----------------------------------------------------
        # Relative depth
        # ----------------------------------------------------

        relative_depth = (
            graph.x[:, 2]
            /
            torch.max(graph.x[:, 2])
        )

        # ----------------------------------------------------
        # Power density
        # ----------------------------------------------------

        col = (
            graph.x[:, 0]
            .numpy()
            .reshape(-1, 1)
        )

        col = std_power_density.transform(
            X=col
        )

        graph.x[:, 0] = torch.tensor(
            col,
            dtype=torch.float32
        ).squeeze(dim=1)

        # ----------------------------------------------------
        # Area
        # ----------------------------------------------------

        col = (
            graph.x[:, 1]
            .numpy()
            .reshape(-1, 1)
        )

        col = np.log1p(col)

        col = std_area.transform(
            X=col
        )

        graph.x[:, 1] = torch.tensor(
            col,
            dtype=torch.float32
        ).squeeze(dim=1)

        # ----------------------------------------------------
        # Kappa
        # ----------------------------------------------------

        col = (
            graph.x[:, 3]
            .numpy()
            .reshape(-1, 1)
        )

        col = np.log1p(col)

        col = std_kappa.transform(
            X=col
        )

        graph.x[:, 3] = torch.tensor(
            col,
            dtype=torch.float32
        ).squeeze(dim=1)

        # ----------------------------------------------------
        # Relative width
        # ----------------------------------------------------

        relative_width = (
            graph.x[:, 10]
            /
            graph.global_features[0, 25]
        )

        # ----------------------------------------------------
        # Relative height
        # ----------------------------------------------------

        relative_height = (
            graph.x[:, 11]
            /
            graph.global_features[0, 26]
        )

        # ----------------------------------------------------
        # Relative x
        # ----------------------------------------------------

        relative_x = (
            graph.x[:, 7]
            /
            graph.global_features[0, 25]
        )

        # ----------------------------------------------------
        # Relative y
        # ----------------------------------------------------

        relative_y = (
            graph.x[:, 8]
            /
            graph.global_features[0, 26]
        )

        # ----------------------------------------------------
        # Exposed area
        # ----------------------------------------------------

        col = (
            graph.x[:, 12]
            .numpy()
            .reshape(-1, 1)
        )

        col = np.log1p(col)

        col = std_exposed_area_mm2.transform(
            X=col
        )

        graph.x[:, 12] = torch.tensor(
            col,
            dtype=torch.float32
        ).squeeze(dim=1)

        # ----------------------------------------------------
        # Final node representation
        # ----------------------------------------------------

        graph.x = torch.concat(
            tensors=[
                graph.x[:, keep],

                encoded_layer_idx,

                encoded_block_type,

                relative_depth.unsqueeze(
                    dim=-1
                ),

                relative_width.unsqueeze(
                    dim=-1
                ),

                relative_height.unsqueeze(
                    dim=-1
                ),

                relative_x.unsqueeze(
                    dim=-1
                ),

                relative_y.unsqueeze(
                    dim=-1
                ),
            ],
            dim=1
        )

        # ====================================================
        # EDGE PREPROCESSING
        # ====================================================

        keep_edge = [
            0,
            1,
            2,
            3,
            4,
            8,
            9,
            10
        ]

        # ----------------------------------------------------
        # Edge type one-hot
        # ----------------------------------------------------

        encoding_edge_type = F.one_hot(
            graph.edge_attr[:, 7].long(),
            Number_of_edge_types
        )

        # ----------------------------------------------------
        # Thermal resistance
        # ----------------------------------------------------

        col = (
            graph.edge_attr[:, 0]
            .numpy()
            .reshape(-1, 1)
        )

        col = np.log1p(col)

        col = std_resistance.transform(
            X=col
        )

        graph.edge_attr[:, 0] = torch.tensor(
            col,
            dtype=torch.float32
        ).squeeze(dim=1)

        # ----------------------------------------------------
        # Geometric distance
        # ----------------------------------------------------

        col = (
            graph.edge_attr[:, 2]
            .numpy()
            .reshape(-1, 1)
        )

        col = std_geometric_distance.transform(
            col
        )

        graph.edge_attr[:, 2] = torch.tensor(
            col,
            dtype=torch.float32
        ).squeeze(dim=1)

        # ----------------------------------------------------
        # Contact overlap area
        # ----------------------------------------------------

        col = (
            graph.edge_attr[:, 4]
            .numpy()
            .reshape(-1, 1)
        )

        col = np.log1p(col)

        col = std_contact_area.transform(
            X=col
        )

        graph.edge_attr[:, 4] = torch.tensor(
            col,
            dtype=torch.float32
        ).squeeze(dim=1)

        # ----------------------------------------------------
        # Thermal path length
        # ----------------------------------------------------

        col = (
            graph.edge_attr[:, 8]
            .numpy()
            .reshape(-1, 1)
        )

        col = np.log1p(col)

        col = std_thermal_path_length.transform(
            X=col
        )

        graph.edge_attr[:, 8] = torch.tensor(
            col,
            dtype=torch.float32
        ).squeeze(dim=1)

        # ----------------------------------------------------
        # Final edge representation
        # ----------------------------------------------------

        graph.edge_attr = torch.concat(
            tensors=[
                graph.edge_attr[:, keep_edge],
                encoding_edge_type
            ],
            dim=1
        )

        # ====================================================
        # GLOBAL PREPROCESSING
        # ====================================================

        KEEP_GLOBAL = [
            0,
            2,
            3,
            5,
            8,
            11,
            12,
            13,
            16,
            19,
            22
        ]

        # ----------------------------------------------------
        # Cooling type one-hot
        # ----------------------------------------------------

        encoding_cooling_types = F.one_hot(
            graph.global_features[:, 7].long(),
            Number_of_cooling_types
        )

        # ----------------------------------------------------
        # Package recipe code one-hot
        # ----------------------------------------------------

        encoding_package_recipe_code = F.one_hot(
            graph.global_features[:, 9].long(),
            Number_of_package_recipe_code
        )

        # ----------------------------------------------------
        # Ambient temperature
        # ----------------------------------------------------

        col = (
            graph.global_features[:, 0]
            .numpy()
            .reshape(-1, 1)
        )

        col = std_ambient_temp_K.transform(
            X=col
        )

        graph.global_features[:, 0] = torch.tensor(
            col,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Workload factor
        # ----------------------------------------------------

        col = (
            graph.global_features[:, 1]
            .numpy()
            .reshape(-1, 1)
        )

        col = std_workload_factor.transform(
            X=col
        )

        graph.global_features[:, 1] = torch.tensor(
            col,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Total power
        # ----------------------------------------------------

        col = (
            graph.global_features[:, 2]
            .numpy()
            .reshape(-1, 1)
        )

        col = np.log1p(col)

        col = std_total_power_W.transform(
            X=col
        )

        graph.global_features[:, 2] = torch.tensor(
            col,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Number of layers
        # ----------------------------------------------------

        graph.global_features[:, 3] = (
            graph.global_features[:, 3]
            /
            torch.tensor(
                data=4
            )
        )

        # ----------------------------------------------------
        # Number of TSV pairs
        # ----------------------------------------------------

        col = (
            graph.global_features[:, 4]
            .numpy()
            .reshape(-1, 1)
        )

        col = np.log1p(col)

        col = std_n_tsv_pairs.transform(
            X=col
        )

        graph.global_features[:, 4] = torch.tensor(
            col,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Hotspot multiplier
        # ----------------------------------------------------

        col = (
            graph.global_features[:, 6]
            .numpy()
            .reshape(-1, 1)
        )

        col = np.log1p(col)

        col = std_hotspot_multiplier.transform(
            X=col
        )

        graph.global_features[:, 6] = torch.tensor(
            col,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Cooling HTC
        # ----------------------------------------------------

        col = (
            graph.global_features[:, 8]
            .numpy()
            .reshape(-1, 1)
        )

        col = np.log1p(col)

        col = std_cooling_htc.transform(
            X=col
        )

        graph.global_features[:, 8] = torch.tensor(
            col,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Bond kappa
        # ----------------------------------------------------

        col = (
            graph.global_features[:, 11]
            .numpy()
            .reshape(-1, 1)
        )

        col = np.log1p(col)

        col = std_bond_kappa_w_mk.transform(
            X=col
        )

        graph.global_features[:, 11] = torch.tensor(
            col,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Bond thickness
        # ----------------------------------------------------

        col = (
            graph.global_features[:, 12]
            .numpy()
            .reshape(-1, 1)
        )

        col = std_bond_thickness_mm.transform(
            X=col
        )

        graph.global_features[:, 12] = torch.tensor(
            col,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Chip width
        # ----------------------------------------------------

        col = (
            graph.global_features[:, 25]
            .numpy()
            .reshape(-1, 1)
        )

        col = std_chip_width_mm.transform(
            X=col
        )

        graph.global_features[:, 25] = torch.tensor(
            col,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Chip height
        # ----------------------------------------------------

        col = (
            graph.global_features[:, 26]
            .numpy()
            .reshape(-1, 1)
        )

        col = std_chip_height_mm.transform(
            X=col
        )

        graph.global_features[:, 26] = torch.tensor(
            col,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Package thermal resistance
        # ----------------------------------------------------

        R_pkg = (

            graph.global_features[:, 14]
            /
            graph.global_features[:, 15].clamp(
                min=1e-8
            )

            +

            graph.global_features[:, 17]
            /
            graph.global_features[:, 18].clamp(
                min=1e-8
            )

            +

            graph.global_features[:, 20]
            /
            graph.global_features[:, 21].clamp(
                min=1e-8
            )

            +

            graph.global_features[:, 23]
            /
            graph.global_features[:, 24].clamp(
                min=1e-8
            )
        )

        # ----------------------------------------------------
        # Final global representation
        # ----------------------------------------------------

        graph.global_features = torch.concat(
            tensors=[

                graph.global_features[
                    :,
                    KEEP_GLOBAL
                ],

                R_pkg.unsqueeze(
                    dim=-1
                ),

                encoding_cooling_types,

                encoding_package_recipe_code,
            ],
            dim=1
        )

        # ====================================================
        # Progress
        # ====================================================

        if (
            graph_index % 250 == 0
            or graph_index == total_graphs - 1
        ):

            print(
                f"Processed graph "
                f"{graph_index + 1}/"
                f"{total_graphs}"
            )

    # ========================================================
    # Verify dimensions
    # ========================================================

    print("\n")
    print("=" * 60)
    print("PREPROCESSING DIMENSION CHECK")
    print("=" * 60)

    first_graph = processed_graphs[0]

    print(
        f"\nNode features: "
        f"{first_graph.x.shape}"
    )

    print(
        f"Edge features: "
        f"{first_graph.edge_attr.shape}"
    )

    print(
        f"Global features: "
        f"{first_graph.global_features.shape}"
    )

    print(
        "\nExpected:"
    )

    print(
        "Node features: 18"
    )

    print(
        "Edge features: 12"
    )

    print(
        "Global features: 21"
    )

    if first_graph.x.shape[1] != 18:

        raise RuntimeError(
            f"\nERROR: Expected 18 node features "
            f"but got {first_graph.x.shape[1]}"
        )

    if first_graph.edge_attr.shape[1] != 11:

        raise RuntimeError(
            f"\nERROR: Expected 12 edge features "
            f"but got {first_graph.edge_attr.shape[1]}"
        )

    if first_graph.global_features.shape[1] != 21:

        raise RuntimeError(
            f"\nERROR: Expected 21 global features "
            f"but got "
            f"{first_graph.global_features.shape[1]}"
        )

    print(
        "\nDimension check passed."
    )

    return processed_graphs


# ============================================================
# MODEL
# ============================================================

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

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Graph convolution layers
        # ----------------------------------------------------

        self.nn_convs = nn.ModuleList()

        self.gat_convs = nn.ModuleList()

        self.merges = nn.ModuleList()

        self.norms = nn.ModuleList()

        for _ in range(num_layers):

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

            self.nn_convs.append(

                geometric_nn.NNConv(
                    hidden_features,
                    hidden_features,
                    edge_mlp,
                    aggr="mean",
                )
            )

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

            self.merges.append(

                nn.Linear(
                    hidden_features * 2,
                    hidden_features
                )
            )

            self.norms.append(

                nn.LayerNorm(
                    hidden_features
                )
            )

        # ----------------------------------------------------
        # Global feature encoder
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Final prediction head
        # ----------------------------------------------------

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
    # Forward
    # ========================================================

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        global_features,
    ):

        h = self.input_proj(x)

        for conv_nn, conv_gat, merge, norm in zip(
            self.nn_convs,
            self.gat_convs,
            self.merges,
            self.norms,
        ):

            residual = h

            h_nn = conv_nn(
                h,
                edge_index,
                edge_attr,
            )

            h_gat = conv_gat(
                h,
                edge_index,
                edge_attr,
            )

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

            h = h + residual

            h = norm(h)

        global_embedding = self.global_encoder(
            global_features
        )

        final_features = torch.cat(
            [
                h,
                global_embedding
            ],
            dim=-1
        )

        return self.FC(
            final_features
        ).squeeze(-1)


# ============================================================
# LOAD DATASET
# ============================================================

def load_graph_list(dataset_path):

    print("\nLoading dataset...")

    obj = torch.load(
        dataset_path,
        map_location="cpu",
        weights_only=False
    )

    # --------------------------------------------------------
    # Case 1: list
    # --------------------------------------------------------

    if isinstance(obj, list):

        return obj

    # --------------------------------------------------------
    # Case 2: dictionary
    # --------------------------------------------------------

    if isinstance(obj, dict):

        for key in (
            "graphs",
            "data_list",
            "dataset",
            "graph_list",
        ):

            if key in obj:

                print(
                    f"Found graph list under key: {key}"
                )

                return obj[key]

        raise RuntimeError(

            "الملف dict لكن مفيش مفتاح معروف جواه.\n"

            f"المفاتيح الموجودة: {list(obj.keys())}\n"

            "عدّل load_graph_list عشان تقرأ المفتاح الصح."
        )

    # --------------------------------------------------------
    # Case 3: Dataset
    # --------------------------------------------------------

    try:

        return list(obj)

    except TypeError:

        raise RuntimeError(

            f"شكل الداتا مش متوقع: {type(obj)}\n"

            "عدّل load_graph_list يدويًا."
        )


# ============================================================
# COMPUTE GRAPH SCORES
# ============================================================

@torch.no_grad()
def compute_graph_scores(
    model,
    processed_graph_list,
    raw_graph_list,
    device,
    target_temp_mean,
    target_temp_scale,
    batch_size=32,
):

    print("\nStarting inference on all graphs...")

    model.eval()

    loader = DataLoader(
        processed_graph_list,
        batch_size=batch_size,
        shuffle=False,
    )

    records = []

    global_idx = 0

    # ========================================================
    # Iterate over batches
    # ========================================================

    for batch_number, batch in enumerate(loader):

        batch = batch.to(device)

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        prediction_normalized = model(

            batch.x,

            batch.edge_index,

            batch.edge_attr,

            batch.global_features[
                batch.batch
            ],
        )

        # ----------------------------------------------------
        # Reverse target normalization
        # ----------------------------------------------------

        log_pred = (

            prediction_normalized
            * target_temp_scale
            + target_temp_mean

        )

        # ----------------------------------------------------
        # Reverse log transformation
        # ----------------------------------------------------

        prediction_kelvin = torch.exp(
            log_pred
        )

        # ----------------------------------------------------
        # Ground truth
        #
        # IMPORTANT:
        # Use RAW y from original graphs.
        # ----------------------------------------------------

        raw_batch = DataLoader(
            [
                raw_graph_list[
                    global_idx + i
                ]
                for i in range(
                    batch.num_graphs
                )
            ],
            batch_size=batch.num_graphs,
            shuffle=False,
        )

        raw_batch = next(
            iter(raw_batch)
        )

        target_kelvin_all = (
            raw_batch.y.reshape(-1)
        )

        # ----------------------------------------------------
        # Since each graph can have a different number
        # of nodes, we cannot directly use batch-local
        # positions without tracking them.
        #
        # Therefore use the processed batch graph IDs.
        # ----------------------------------------------------

        node_graph_id = batch.batch

        num_graphs = batch.num_graphs

        # ====================================================
        # Process each graph
        # ====================================================

        node_offset = 0

        for g in range(num_graphs):

            mask = (
                node_graph_id == g
            )

            pred_g = prediction_kelvin[
                mask
            ]

            num_nodes_g = int(
                mask.sum().item()
            )

            target_g = target_kelvin_all[
                node_offset:
                node_offset + num_nodes_g
            ].to(device)

            node_offset += num_nodes_g

            # ------------------------------------------------
            # Error
            # ------------------------------------------------

            error = (
                pred_g - target_g
            )

            bias = error.mean().item()

            mae = error.abs().mean().item()

            rmse = torch.sqrt(
                (error ** 2).mean()
            ).item()

            # ------------------------------------------------
            # Graph record
            # ------------------------------------------------

            record = {

                "dataset_index": global_idx,

                "num_nodes": num_nodes_g,

                "bias_K": bias,

                "abs_bias_K": abs(bias),

                "mae_K": mae,

                "rmse_K": rmse,

                "mean_target_K":
                    target_g.mean().item(),

                "min_target_K":
                    target_g.min().item(),

                "max_target_K":
                    target_g.max().item(),
            }

            # ------------------------------------------------
            # Save RAW global features
            # ------------------------------------------------

            g_feats = (
                raw_graph_list[
                    global_idx
                ]
                .global_features
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            for i, val in enumerate(g_feats):

                record[
                    f"global_feat_{i}"
                ] = float(val)

            records.append(
                record
            )

            global_idx += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_number % 10 == 0
            or batch_number == len(loader) - 1
        ):

            print(

                f"Processed batch "
                f"{batch_number + 1}/"
                f"{len(loader)} "

                f"| Graphs processed: "
                f"{global_idx}"
            )

    return records


# ============================================================
# SELECT HARD GRAPHS
# ============================================================

def select_hard_graphs(
    records,
    abs_bias_threshold_K=8.0,
    top_percentile=0.10,
):

    sorted_by_abs_bias = sorted(

        records,

        key=lambda r:
            r["abs_bias_K"],

        reverse=True,
    )

    n_top = max(

        1,

        int(
            len(sorted_by_abs_bias)
            * top_percentile
        ),
    )

    top_percentile_set = {

        r["dataset_index"]

        for r in
        sorted_by_abs_bias[:n_top]
    }

    threshold_set = {

        r["dataset_index"]

        for r in records

        if r["abs_bias_K"]
        >= abs_bias_threshold_K
    }

    selected_indices = (

        top_percentile_set
        |
        threshold_set
    )

    selected_records = [

        r

        for r in records

        if r["dataset_index"]
        in selected_indices
    ]

    print("\n")
    print("=" * 60)
    print("HARD GRAPH SELECTION")
    print("=" * 60)

    print(

        f"\n|bias| >= "
        f"{abs_bias_threshold_K} K:"
        f" {len(threshold_set)} graphs"
    )

    print(

        f"Top {top_percentile * 100:.0f}% "
        f"by |bias|:"
        f" {len(top_percentile_set)} graphs"
    )

    print(

        f"Total after UNION:"
        f" {len(selected_records)} graphs"
    )

    print("=" * 60)

    return selected_records


# ============================================================
# MAIN
# ============================================================

def main(
    dataset_path,
    checkpoint_path,
    output_path,
    abs_bias_threshold_K=8.0,
    top_percentile=0.10,
    batch_size=32,
):

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("HARD GRAPH EXTRACTION")
    print("=" * 60)

    print(
        f"\nDevice: {device}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    print(
        f"\nDataset path:\n"
        f"{dataset_path}"
    )

    graph_list = load_graph_list(
        dataset_path
    )

    print(
        f"\nTotal number of graphs: "
        f"{len(graph_list)}"
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    (
        Train_Nodes_scallers,
        Train_Edges_scallers,
        Train_global_scallers,
        Train_one_hot_Nodes_data,
        Train_one_hot_Edges_data,
        Train_one_hot_global_data,
        normalized_temp_scaler,
    ) = load_preprocessing_objects()

    processed_graph_list = preprocess_graph_list(

        graph_list,

        Train_Nodes_scallers,

        Train_Edges_scallers,

        Train_global_scallers,

        Train_one_hot_Nodes_data,

        Train_one_hot_Edges_data,

        Train_one_hot_global_data,
    )

    # ========================================================
    # Checkpoint
    # ========================================================

    print(
        f"\nCheckpoint path:\n"
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device,

        weights_only=False,
    )

    saved_config = checkpoint.get(
        "config",
        CONFIG
    )

    print(
        "\nBuilding model..."
    )

    # ========================================================
    # Build model
    # ========================================================

    model = ML_only_model(

        in_features=saved_config[
            "in_features"
        ],

        hidden_features=saved_config[
            "hidden_features"
        ],

        in_features_edge=saved_config[
            "in_features_edge"
        ],

        global_features=saved_config[
            "global_features"
        ],

        num_layers=saved_config[
            "num_layers"
        ],

        edge_mlp_hidden=saved_config[
            "edge_mlp_hidden"
        ],

        gat_heads=saved_config[
            "gat_heads"
        ],
    ).to(device)

    # ========================================================
    # Load checkpoint
    # ========================================================

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "\nModel loaded successfully."
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', '?')}"
    )

    print(
        f"Saved RMSE: "
        f"{checkpoint.get('rmse', '?')}"
    )

    # ========================================================
    # Target scaler
    # ========================================================

    std_target_temp = (
        normalized_temp_scaler[
            "std_target_temp"
        ]
    )

    target_temp_mean = torch.tensor(

        float(
            np.asarray(
                std_target_temp.mean_
            )
            .reshape(-1)[0]
        ),

        dtype=torch.float32,

        device=device,
    )

    target_temp_scale = torch.tensor(

        float(
            np.asarray(
                std_target_temp.scale_
            )
            .reshape(-1)[0]
        ),

        dtype=torch.float32,

        device=device,
    )

    print(
        f"\nTarget scaler mean: "
        f"{target_temp_mean.item()}"
    )

    print(
        f"Target scaler scale: "
        f"{target_temp_scale.item()}"
    )

    # ========================================================
    # Compute graph scores
    # ========================================================

    print("\n")
    print("=" * 60)
    print("COMPUTING GRAPH ERRORS")
    print("=" * 60)

    records = compute_graph_scores(

        model,

        processed_graph_list,

        graph_list,

        device,

        target_temp_mean,

        target_temp_scale,

        batch_size=batch_size,
    )

    print(
        f"\nFinished inference."
    )

    print(
        f"Total records: "
        f"{len(records)}"
    )

    # ========================================================
    # Select hard graphs
    # ========================================================

    selected_records = select_hard_graphs(

        records,

        abs_bias_threshold_K=
            abs_bias_threshold_K,

        top_percentile=
            top_percentile,
    )

    # ========================================================
    # Get selected indices
    # ========================================================

    selected_indices = [

        r["dataset_index"]

        for r in selected_records
    ]

    # ========================================================
    # Extract RAW graphs
    # ========================================================

    hard_graphs = [

        graph_list[i]

        for i in selected_indices
    ]

    # ========================================================
    # Save hard graphs
    # ========================================================

    torch.save(

        hard_graphs,

        output_path
    )

    print(

        f"\nSaved "
        f"{len(hard_graphs)} hard graphs to:"
    )

    print(
        output_path
    )

    # ========================================================
    # Save metrics CSV
    # ========================================================

    df = pd.DataFrame(
        selected_records
    )

    df = df.sort_values(

        "abs_bias_K",

        ascending=False,
    )

    if output_path.lower().endswith(
        ".pt"
    ):

        csv_path = (

            output_path[:-3]

            + "_metrics.csv"
        )

    else:

        csv_path = (

            output_path

            + "_metrics.csv"
        )

    df.to_csv(

        csv_path,

        index=False,
    )

    print(
        f"\nMetrics CSV saved to:"
    )

    print(
        csv_path
    )

    # ========================================================
    # Summary
    # ========================================================

    print("\n")
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    print(

        f"\nOriginal graphs:"
        f" {len(graph_list)}"
    )

    print(

        f"Selected hard graphs:"
        f" {len(hard_graphs)}"
    )

    print(
        f"\nHard graphs file:"
    )

    print(
        output_path
    )

    print(
        f"\nMetrics file:"
    )

    print(
        csv_path
    )

    # ========================================================
    # Training usage
    # ========================================================

    print("\n")
    print("=" * 60)
    print("USAGE IN TRAINING")
    print("=" * 60)

    print(
        "\nExample:"
    )

    print(
        "hard_graphs = torch.load("
        f"'{output_path}', "
        "weights_only=False"
        ")"
    )

    print(
        "\n# Repeat them N times:"
    )

    print(
        "oversampled_train_list = "
        "original_train_list + "
        "hard_graphs * N"
    )

    print(
        "\n# Then create your DataLoader normally."
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main(

        dataset_path=DATASET_PATH,

        checkpoint_path=CHECKPOINT_PATH,

        output_path=OUTPUT_PATH,

        abs_bias_threshold_K=
            ABS_BIAS_THRESHOLD_K,

        top_percentile=
            TOP_PERCENTILE,

        batch_size=
            BATCH_SIZE,
    )
