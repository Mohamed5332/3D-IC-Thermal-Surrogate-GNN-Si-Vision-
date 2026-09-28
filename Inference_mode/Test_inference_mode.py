import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from Extract_ground_truth_temperatures import build_ground_truth
from convert_user_inputs_to_h5 import parse_stk_file, parse_flp_file
from Extract_ground_truth_temperatures import build_ground_truth_in_graph_order


from Preprocessing.process_one_graph import Train_Nodes_data_pipleline , Train_Edges_data_pipleline , Train_Global_data_pipleline_new , Train_Global_data_pipleline 

from Stack_floorplan_to_graph import build_one_graph
import torch
from Backend.Final_app import model , device
import matplotlib.pyplot as plt
from joblib import load , dump
from Plotting_results.plot_predictionsVSGroundTruth import plot_graph

model.eval()

input_dir = "C:/Users/dell/Downloads/handoff_batch50-20260915T111020Z-1-001/handoff_batch50/sim_900056_RAW/sim_900056"

graph = build_one_graph(input_dir)

ground_truth_temperatures = build_ground_truth_in_graph_order(input_dir)

print("ground_truth_temperatures:", ground_truth_temperatures)
print("ground truth count:", len(ground_truth_temperatures))


print("sim_id:", graph.sim_id)
print("nodes:", graph.total_node_count, "(real=", graph.real_node_count, ", filler=", graph.filler_node_count, ")")
print("edges:", graph.edge_index.shape[1])
print("x:", tuple(graph.x.shape))
print("edge_attr:", tuple(graph.edge_attr.shape))
print("global_features:", tuple(graph.global_features.shape))
print("temperature :" , ground_truth_temperatures)


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


    result_list = result.squeeze().detach().cpu().tolist()


    graph.y = torch.tensor(
        ground_truth_temperatures,
        dtype=torch.float32
    )

    print("result : " , result_list)
    print("ground truth temperature : " , ground_truth_temperatures)

    print("prediction count:", len(result_list))
    print("ground truth : " , len(ground_truth_temperatures))


    plot_graph(graph= graph , result= result)
    plt.show()