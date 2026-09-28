
print("Final app in action...")
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
#ML_with_physics_model_ultra_super_power_V2000.pt (production...)
#Model_v1_playground.pt (production level)
#Model_v2_playground.pt (production level)
#Model_v3_playground.pt (production level)
#Model_v4_playground.pt (production level)
#Model_v7_playground.pt (production level) ("Best one till now....")
    # ---- checkpoint ----
    "checkpoint_path": r"C:/msys64/home/dell/Model_v7_playground.pt",

    # ---- Jensen correction ----
    "use_lognormal_correction": True,
}


device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
                geometric_nn.NNConv(hidden_features, hidden_features, edge_mlp, aggr="mean")
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


normalized_temp_scaler = load(filename="normalized_temp_scaller")

std_target_temp = normalized_temp_scaler["std_target_temp"]

target_temp_mean = torch.tensor(
    std_target_temp.mean_[0],
    dtype=torch.float32,
    device=device
)

target_temp_scale = torch.tensor(std_target_temp.scale_[0] , dtype=torch.float32 , device=device)


print("Model : " , model)



"""
model.eval()



# This graph was preprocessed already...
for batch in loaders['test']:
    graph = batch[0]
    break

print(graph.x.shape)
print(graph.edge_index.shape)
print(graph.edge_attr.shape)
print(graph.global_features.shape)
print(target_temp_mean)
print(target_temp_scale)


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

    
    #residual_log_var = torch.tensor(
    #    checkpoint["residual_log_var"],
    #    dtype=torch.float32,
    #    device=device
    #)
    
    result = torch.exp(output * target_temp_scale + target_temp_mean).unsqueeze(dim=1)


    print("\nPREDICTION vs GROUND TRUTH")
    print("=" * 60)

    real_count = len(graph.y)

    for i in range(real_count):
        pred = result[i].item()
        true = graph.y[i].item()
        error = abs(pred - true)

        print(
            f"Node {i:2d} | "
            f"Prediction = {pred:8.4f} K | "
            f"Ground Truth = {true:8.4f} K | "
            f"Error = {error:7.4f} K"
        )


    plot_graph(graph, result)
    plt.show()
"""