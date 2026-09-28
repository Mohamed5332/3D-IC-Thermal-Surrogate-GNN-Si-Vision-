print("FASTAPI app in action...")
from fastapi.responses import FileResponse

from Inference_mode.Stack_floorplan_to_graph import build_one_graph
import torch
from Preprocessing.process_one_graph import Train_Nodes_data_pipleline , Train_Edges_data_pipleline , Train_Global_data_pipleline_new , Train_Global_data_pipleline 

import uuid
from click import File 
import torch
from joblib import load
import torch.nn as nn 
import torch_geometric.nn as geometric_nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Plotting_results.plot_predictionsVSGroundTruth import plot_graph
from fastapi import FastAPI, UploadFile


from fastapi import FastAPI, UploadFile, File
from typing import List
import tempfile
import os
import shutil

from pathlib import Path
from joblib import load

ARTIFACT_DIR = Path(r"C:/msys64/home/dell")
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

CONFIG = {
    # ---- architecture ----
    "in_features":       18,
    "in_features_edge":  11,
    "global_features":   21,
    "hidden_features":   64,
    "edge_mlp_hidden":   32,
    "num_layers":        4,
    "gat_heads":         2,

    # ---- checkpoint ----
    "checkpoint_path": r"C:/msys64/home/dell/ML_with_physics_model_ultra_super_power_V2000.pt",

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



model = ML_only_model(in_features=CONFIG['in_features'] , 
                      hidden_features= CONFIG['hidden_features'] , 
                      in_features_edge= CONFIG['in_features_edge'] , 
                      global_features=CONFIG['global_features'] , 
                      num_layers=CONFIG['num_layers'] , 
                      edge_mlp_hidden=CONFIG['edge_mlp_hidden'] , 
                      gat_heads=2).to(device)

model.load_state_dict(state_dict= checkpoint['model_state_dict'])


model.eval()


normalized_temp_scaler = load(
    filename= ARTIFACT_DIR / "normalized_temp_scaller"
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


# Add the rest here....
app = FastAPI()


async def save_uploaded_folder(Files: list[UploadFile]) -> str :
    input_dir =  tempfile.mkdtemp(prefix="thermal_")
    for file in Files:
        file_name = os.path.basename(file.filename)

        file_path = os.path.join(input_dir , file_name)

        with open(file_path , "wb") as buffer :
            shutil.copyfileobj(file.file , buffer)

    return input_dir        



@app.get("/plot/{plot_name}")
async def get_plot(plot_name: str):

    plot_path = os.path.join(PLOT_DIR , plot_name)
    
    return FileResponse(plot_path)




@app.post("/predict")
async def predict(Files: list[UploadFile] = File(...)):

    input_dir = await save_uploaded_folder(Files= Files)

    graph = build_one_graph(input_dir)

    graph = Train_Nodes_data_pipleline(graph= graph)
    graph = Train_Edges_data_pipleline(graph= graph)
    graph = Train_Global_data_pipleline_new(graph= graph)


    data = {
    'input_features' : graph.x,
    'edge_index' : graph.edge_index ,
    'edge_attr' : graph.edge_attr,
    'global_features' : graph.global_features.expand(graph.x.size(0), -1)
    }
    with torch.no_grad():
        output = model(data['input_features'].to(device) , 
                    data['edge_index'].to(device) , 
                    data['edge_attr'].to(device) , 
                    data['global_features'].to(device))

    result = torch.exp( output * target_temp_scale + target_temp_mean ).unsqueeze(dim=1)

    plot_name = f"prediction_{uuid.uuid4().hex}.png"
    plot_path = os.path.join(PLOT_DIR , plot_name)


    plot_graph(graph, result)
    
    plt.savefig(plot_path , bbox_inches="tight")

    plt.close()
    
    return {
        "status": "success",
        "num_nodes": result.shape[0],
        "min_temperature_K": result.min().item(),
        "max_temperature_K": result.max().item(),
        "mean_temperature_K": result.mean().item(),
        "temperatures_K": result.squeeze(1).cpu().tolist(),
        "plot_name": plot_name
    }

