import torch 
import numpy as np
from pprint import pprint as pprint
import torch.nn.functional as F
from joblib import dump , load


# scallers.
val_Nodes_scallers = load(filename="scallers_Node_data")
val_Edges_scallers = load(filename="scallers_Edege_data")
val_global_scallers = load(filename="scallers_global_data")


# meta data of one hot encoding.
val_one_hot_Nodes_data = load(filename="one_hot_Node_data")
val_one_hot_Edges_data = load(filename="one_hot_edge_data")
val_one_hot_global_data = load(filename="one_hot_global_data")



# load scallers and meta_data of one hot encoding.(Nodes)
Number_of_layers = val_one_hot_Nodes_data['Number_of_layers']
Number_of_block_types = val_one_hot_Nodes_data['Number_of_block_types']

std_power_density = val_Nodes_scallers['std_power_density']
std_area = val_Nodes_scallers['std_area']
std_kappa = val_Nodes_scallers['std_kappa']
std_exposed_area_mm2 = val_Nodes_scallers['std_exposed_area_mm2']


# load scallers and meta_data of one hot encoding.(Edges)
Number_of_edge_types = val_one_hot_Edges_data['Number_of_edge_types'] 

std_resistance = val_Edges_scallers['std_resistance']
std_geometric_distance = val_Edges_scallers['std_geometric_distance']
std_contact_area = val_Edges_scallers['std_contact_area']
std_thermal_path_length = val_Edges_scallers['std_thermal_path_length']



# load scallers and meta_data of one hot encoding.(global)
Number_of_cooling_types = val_one_hot_global_data['Number_of_cooling_types']
Number_of_package_recipe_code = val_one_hot_global_data['Number_of_package_recipe_code']


std_total_power_W = val_global_scallers['std_total_power_W']
std_cooling_htc = val_global_scallers['std_cooling_htc']
std_bond_kappa_w_mk = val_global_scallers['std_bond_kappa_w_mk']
std_bond_thickness_mm = val_global_scallers['std_bond_thickness_mm']

std_ambient_temp_K = val_global_scallers['std_ambient_temp_K']
std_workload_factor = val_global_scallers['std_workload_factor']
std_n_tsv_pairs = val_global_scallers['std_n_tsv_pairs']
std_hotspot_multiplier = val_global_scallers['std_hotspot_multiplier']
std_chip_width_mm = val_global_scallers['std_chip_width_mm']
std_chip_height_mm = val_global_scallers['std_chip_height_mm']



# load scaller of normalized Temperature.
normalized_temp_scaller = load(filename="normalized_temp_scaller")
std_target_temp = normalized_temp_scaller['std_target_temp']

def Val_Nodes_data_pipleline(Val_data):
    # Start preprocessing loop for Node Features.
    keep = [0 , 1 , 3 , 5 , 6 , 12]
    for graph in Val_data:

        # one hot encoding of layer_idx and block type.
        encoded_layer_idx = F.one_hot(graph.x[: , 2].long() , Number_of_layers)
        encoded_block_type = F.one_hot(graph.x[: , 4].long() , Number_of_block_types) 
        
        # relative depth (New Feature).
        relative_depth = graph.x[: ,2] / torch.max(graph.x[: ,2])

        # handle power.
        col = graph.x[:,0].numpy().reshape(-1, 1)
        col = std_power_density.transform(X = col)
        graph.x[:,0] = torch.tensor(col , dtype = torch.float32).squeeze(dim = 1)

        # handle area
        col = graph.x[: , 1].numpy().reshape(-1, 1)
        col = np.log1p(col)
        col = std_area.transform(X = col)
        graph.x[: , 1] = torch.tensor(col , dtype = torch.float32).squeeze(dim = 1)

        # handle Kappa.
        col = graph.x[: , 3].numpy().reshape(-1, 1)
        col = np.log1p(col)
        col = std_kappa.transform(X = col)
        graph.x[: , 3] = torch.tensor(col , dtype = torch.float32).squeeze(dim = 1)

        # handle width.
        relative_width = graph.x[: , 10] / graph.global_features[0 , 25]

        # handle height.
        relative_height = graph.x[:,11] / graph.global_features[0, 26]

        # Add relative x , y.
        relative_x = graph.x[: , 7] / graph.global_features[0 , 25]
        relative_y = graph.x[: , 8] / graph.global_features[0 , 26]


        # handle exposed area
        col = graph.x[: , 12].numpy().reshape(-1, 1)
        col = np.log1p(col)
        col = std_exposed_area_mm2.transform(X = col)
        graph.x[: , 12] = torch.tensor(col , dtype = torch.float32).squeeze(dim = 1)

        # Concat Data
        graph.x = torch.concat(tensors= [graph.x[: , keep] , encoded_layer_idx , encoded_block_type , relative_depth.unsqueeze(dim=-1) 
                                        , relative_width.unsqueeze(dim=-1) , relative_height.unsqueeze(dim=-1) 
                                        , relative_x.unsqueeze(dim=-1) , relative_y.unsqueeze(dim=-1)] , dim=1)

    return Val_data
   




def Val_Edges_data_pipleline(Val_data):
    # start preporcessing loop of edge features.
    keep = [0,1,2,3,4,8,9,10]
    for graph in Val_data:

        # one hot encoding (Edge type).
        encoding_edge_type = F.one_hot(graph.edge_attr[: , 7].long() , Number_of_edge_types)
        

        # handle thermal resistance.
        col = graph.edge_attr[: , 0].numpy().reshape(-1 , 1)
        col = np.log1p(col)
        col = std_resistance.transform(X = col)
        graph.edge_attr[ : , 0] = torch.tensor(data = col , dtype = torch.float32).squeeze(dim=1)


        # handle geometric distance.
        col = graph.edge_attr[: , 2].numpy().reshape(-1 , 1)
        col = std_geometric_distance.transform(col)
        graph.edge_attr[ : , 2] = torch.tensor(data = col , dtype = torch.float32).squeeze(dim = 1)


        # handle_contanct_overlap_area
        col = graph.edge_attr[ : , 4].numpy().reshape(-1, 1)
        col = np.log1p(col)
        col = std_contact_area.transform(X = col)
        graph.edge_attr[: , 4] = torch.tensor(data = col , dtype = torch.float32).squeeze(dim = 1)


        # handle thermal_path_length 
        col = graph.edge_attr[: , 8].numpy().reshape(-1, 1)
        col = np.log1p(col)
        col = std_thermal_path_length.transform(X = col)
        graph.edge_attr[: , 8] = torch.tensor(data = col ,dtype= torch.float32).squeeze(dim = 1)


        # Concat data .
        graph.edge_attr = torch.concat(tensors = [graph.edge_attr[: , keep] , encoding_edge_type] , dim = 1)


    return Val_data    





def Val_Global_data_pipleline(Val_data):
    KEEP_GLOBAL = [0, 2, 3, 8, 11, 12]
    for graph in Val_data:

        # one hot encoding for cooling types
        encoding_cooling_types = F.one_hot(graph.global_features[: , 7].long() , Number_of_cooling_types)


        # Handle total power.
        col = graph.global_features[: ,2].numpy().reshape(-1,1)
        col = np.log1p(col)
        col = std_total_power_W.transform(X=col)
        graph.global_features[:, 2] = torch.tensor(data= col , dtype=torch.float32)


        # handle the N_layers.
        graph.global_features[: , 3] = graph.global_features[: , 3] / torch.tensor(data=4)

        # handle cooling_htc_w_m2k
        col = graph.global_features[: , 8].numpy().reshape(-1,1)
        col = np.log1p(col)
        col = std_cooling_htc.transform(X= col)
        graph.global_features[: , 8] = torch.tensor(data= col , dtype=torch.float32)


        # handle bond kappa
        col = graph.global_features[: , 11].numpy().reshape(-1,1)
        col = np.log1p(col)
        col = std_bond_kappa_w_mk.transform(X= col)
        graph.global_features[: , 11] = torch.tensor(data= col , dtype=torch.float32)


        # handle std_bond_thickness_mm
        col = graph.global_features[: , 12].numpy().reshape(-1,1)
        col = std_bond_thickness_mm.transform(X = col)
        graph.global_features[: , 12] = torch.tensor(data = col , dtype = torch.float32)

        R_pkg = ( graph.global_features[: , 14] / graph.global_features[: , 15].clamp(min=1e-8)      # RDL
            + graph.global_features[: , 17] / graph.global_features[: , 18].clamp(min=1e-8)      # Interposer
            + graph.global_features[ : , 20] / graph.global_features[: , 21].clamp(min=1e-8)      # C4
            + graph.global_features[: , 23] / graph.global_features[: , 24].clamp(min=1e-8))    # Substrate


        # Concat Data.
        graph.global_features = torch.concat(tensors= [graph.global_features[: , KEEP_GLOBAL] , R_pkg.unsqueeze(dim = -1) , encoding_cooling_types] , dim = 1)

    return Val_data



def val_Global_data_pipleline_new(val_data):
    KEEP_GLOBAL = [0, 2, 3 , 5, 8, 11, 12 , 13 , 16 , 19 , 22 ]
    for graph in val_data:

        # one hot encoding for cooling types
        encoding_cooling_types = F.one_hot(graph.global_features[: , 7].long() , Number_of_cooling_types)

        #one hot encoding for package_recipe_code
        encoding_package_recipe_code = F.one_hot(graph.global_features[: , 9].long() , Number_of_package_recipe_code)

        
        # handle ambient_temp_K (new).
        col = graph.global_features[: ,0].numpy().reshape(-1,1)
        col = std_ambient_temp_K.transform(X=col)
        graph.global_features[:, 0] = torch.tensor(data= col , dtype=torch.float32)


        # handle workload_factor (new).
        col = graph.global_features[: ,1].numpy().reshape(-1,1)
        col = std_workload_factor.transform(X=col)
        graph.global_features[:, 1] = torch.tensor(data= col , dtype=torch.float32)

        
        # Handle total power.
        col = graph.global_features[: ,2].numpy().reshape(-1,1)
        col = np.log1p(col)
        col = std_total_power_W.transform(X=col)
        graph.global_features[:, 2] = torch.tensor(data= col , dtype=torch.float32)


        # handle the N_layers.
        graph.global_features[: , 3] = graph.global_features[: , 3] / torch.tensor(data=4)

        
        # Handle n_tsv_pairs (new).
        col = graph.global_features[: ,4].numpy().reshape(-1,1)
        col = np.log1p(col)
        col = std_n_tsv_pairs.transform(X=col)
        graph.global_features[:, 4] = torch.tensor(data= col , dtype=torch.float32)



        #handle hotspot_multiplier (new).
        col = graph.global_features[: ,6].numpy().reshape(-1,1)
        col = np.log1p(col)
        col = std_hotspot_multiplier.transform(X=col)
        graph.global_features[:, 6] = torch.tensor(data= col , dtype=torch.float32)
        

        # handle cooling_htc_w_m2k
        col = graph.global_features[: , 8].numpy().reshape(-1,1)
        col = np.log1p(col)
        col = std_cooling_htc.transform(X= col)
        graph.global_features[: , 8] = torch.tensor(data= col , dtype=torch.float32)


        # handle bond kappa
        col = graph.global_features[: , 11].numpy().reshape(-1,1)
        col = np.log1p(col)
        col = std_bond_kappa_w_mk.transform(X= col)
        graph.global_features[: , 11] = torch.tensor(data= col , dtype=torch.float32)


        # handle std_bond_thickness_mm
        col = graph.global_features[: , 12].numpy().reshape(-1,1)
        col = std_bond_thickness_mm.transform(X = col)
        graph.global_features[: , 12] = torch.tensor(data = col , dtype = torch.float32)

        # handle chip_width_mm (new)
        col = graph.global_features[: , 25].numpy().reshape(-1,1)
        col = std_chip_width_mm.transform(X = col)
        graph.global_features[: , 25] = torch.tensor(data = col , dtype = torch.float32)

        # handle chip_height_mm (new)
        col = graph.global_features[: , 26].numpy().reshape(-1,1)
        col = std_chip_height_mm.transform(X = col)
        graph.global_features[: , 26] = torch.tensor(data = col , dtype = torch.float32)
        
        # 13 , 16 , 19 , 22 , 25 , 26
        R_pkg = ( graph.global_features[: , 14] / graph.global_features[: , 15].clamp(min=1e-8)      # RDL
            + graph.global_features[: , 17] / graph.global_features[: , 18].clamp(min=1e-8)      # Interposer
            + graph.global_features[ : , 20] / graph.global_features[: , 21].clamp(min=1e-8)      # C4
            + graph.global_features[: , 23] / graph.global_features[: , 24].clamp(min=1e-8))    # Substrate


        # Concat Data.
        graph.global_features = torch.concat(tensors= [graph.global_features[: , KEEP_GLOBAL] , R_pkg.unsqueeze(dim = -1) , encoding_cooling_types , encoding_package_recipe_code] , dim = 1)

    return val_data

def val_normalize_temp(val_data):
    for graph in val_data:
        col = graph.y.numpy().reshape(-1,1)
        col = np.log(col)
        col = std_target_temp.transform(X=col)
        graph.T_real_normalized = torch.tensor(data = col , dtype = torch.float32).squeeze(dim = 1)

    return val_data