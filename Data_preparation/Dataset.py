import torch 
import pandas as pd 
import numpy as np
from pprint import pprint as pprint
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from joblib import dump , load
from torch_geometric.data import DataLoader 

# Here i will add the distribution part here...
# Load data.
#data = torch.load("/home/share/ml-repo/real_fea_delivery/thermal_graph_v2_dataset.pt")


# ============================================================
# 1. LOAD DATASET
# ============================================================

attributes_to_remove = ["source_file", "filler_geometry_source", "filler_blocks_width",
                        "real_geometry_source", "block_features_merged_width",
                        "block_features_width", "schema_version",
                        "tsv_pads", "tsv_pad_count", "tsv_pads_used_as_nodes",
                        "tsv_pads_used_in_resistance" ]


from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent



dataset_physics  = "D:/Thermal surrogate project/Datasets/thermal_graph_PINN_V4_2_balanced_audited (1).pt"

# new...
dataset_physics1 = "D:/Thermal surrogate project/Datasets/thermal_graph_filtered_450_500K_v2schema.pt"
dataset_physics2 = "D:/Thermal surrogate project/Datasets/thermal_graph_phase2d_mate_run400_500_v2schema.pt"
dataset_physics3 = "D:/Thermal surrogate project/Datasets/thermal_graph_phase2d_mate_400_500K_ONLY_v2schema.pt"


# new...
dataset = torch.load(f = dataset_physics , weights_only = False)
new_real_400_500k_datasetv1 = torch.load(f= dataset_physics2 , weights_only = False)
new_real_400_500k_datasetv2 = torch.load(f= dataset_physics3 , weights_only = False)
new_real_450_500k_datasetv3 = torch.load(f= dataset_physics1 , weights_only= False)

# new...
dataset = dataset + new_real_400_500k_datasetv1 + new_real_400_500k_datasetv2

def remove_graphs_outside_temperature_range(dataset, min_temp=300.0, max_temp=500.0):
    filtered_dataset = []
    removed_count = 0

    for graph in dataset:

        # Only real/valid nodes
        valid_temps = graph.y[graph.target_mask]

        # If ANY node is outside [300, 500], remove the whole graph
        if ((valid_temps < min_temp) | (valid_temps > max_temp)).any():
            removed_count += 1
            continue

        filtered_dataset.append(graph)

    print(f"Original graphs : {len(dataset)}")
    print(f"Removed graphs  : {removed_count}")
    print(f"Remaining graphs: {len(filtered_dataset)}")

    return filtered_dataset



dataset = remove_graphs_outside_temperature_range(dataset= dataset , min_temp= 300 , max_temp= 500)


# ============================================================
# 2. SETTINGS
# ============================================================

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

BATCH_SIZE = 16

# عدل الـ bins حسب الـ temperature range الحقيقي عندك
bins = [300, 350, 400, 450, 500]


# ============================================================
# 3. CREATE GRAPH INDICES
# ============================================================

indices = np.arange(len(dataset))

print("Total indices:", len(indices))


# ============================================================
# 4. FUNCTION TO GET TEMPERATURES
# ============================================================

def get_temperatures(dataset, graph_indices):

    temperatures = []

    for idx in graph_indices:

        T = dataset[idx].y.detach().cpu().numpy().reshape(-1)

        temperatures.extend(T)

    return np.array(temperatures)


# ============================================================
# 5. FUNCTION TO CALCULATE TEMPERATURE DISTRIBUTION
# ============================================================

def get_distribution(temperatures, bins):

    hist, _ = np.histogram(
        temperatures,
        bins=bins
    )

    distribution = hist / hist.sum() * 100

    return distribution


# ============================================================
# 6. SEARCH FOR BEST RANDOM STATE
# ============================================================

best_seed = None
best_score = float("inf")

best_train_idx = None
best_val_idx = None
best_test_idx = None


for seed in range(100):

    # --------------------------------------------------------
    # Train = 70%
    # Temporary = 30%
    # --------------------------------------------------------

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        shuffle=True
    )

    # --------------------------------------------------------
    # Val = 15%
    # Test = 15%
    # --------------------------------------------------------

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=seed,
        shuffle=True
    )

    # --------------------------------------------------------
    # Get temperatures
    # --------------------------------------------------------

    train_T = get_temperatures(
        dataset,
        train_idx
    )

    val_T = get_temperatures(
        dataset,
        val_idx
    )

    test_T = get_temperatures(
        dataset,
        test_idx
    )

    # --------------------------------------------------------
    # Get distributions
    # --------------------------------------------------------

    train_dist = get_distribution(
        train_T,
        bins
    )

    val_dist = get_distribution(
        val_T,
        bins
    )

    test_dist = get_distribution(
        test_T,
        bins
    )

    # --------------------------------------------------------
    # Calculate differences
    # --------------------------------------------------------

    train_val_diff = np.mean(
        np.abs(
            train_dist -
            val_dist
        )
    )

    train_test_diff = np.mean(
        np.abs(
            train_dist -
            test_dist
        )
    )

    val_test_diff = np.mean(
        np.abs(
            val_dist -
            test_dist
        )
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    score = (
        train_val_diff +
        train_test_diff +
        val_test_diff
    )

    # --------------------------------------------------------
    # Save best split
    # --------------------------------------------------------

    if score < best_score:

        best_score = score

        best_seed = seed

        best_train_idx = train_idx
        best_val_idx = val_idx
        best_test_idx = test_idx


# ============================================================
# 7. PRINT BEST SPLIT
# ============================================================

print("\n========================================")
print("BEST SPLIT")
print("========================================")

print("Best random state:", best_seed)

print("Train graphs:", len(best_train_idx))
print("Val graphs:", len(best_val_idx))
print("Test graphs:", len(best_test_idx))

print("Distribution score:", best_score)


# ============================================================
# 8. GET FINAL TEMPERATURE DISTRIBUTIONS
# ============================================================

train_T = get_temperatures(
    dataset,
    best_train_idx
)

val_T = get_temperatures(
    dataset,
    best_val_idx
)

test_T = get_temperatures(
    dataset,
    best_test_idx
)


train_dist = get_distribution(
    train_T,
    bins
)

val_dist = get_distribution(
    val_T,
    bins
)

test_dist = get_distribution(
    test_T,
    bins
)


# ============================================================
# 9. PRINT FINAL DISTRIBUTION
# ============================================================

print("\n========================================")
print("TEMPERATURE DISTRIBUTION")
print("========================================")

for i in range(len(bins) - 1):

    print(
        f"{bins[i]}-{bins[i+1]} K : "
        f"Train = {train_dist[i]:.2f}% | "
        f"Val = {val_dist[i]:.2f}% | "
        f"Test = {test_dist[i]:.2f}%"
    )


# ============================================================
# 10. CREATE FINAL DATASETS
# ============================================================

train_data = [
    dataset[i]
    for i in best_train_idx
]

val_data = [
    dataset[i]
    for i in best_val_idx
]

test_data = [
    dataset[i]
    for i in best_test_idx
]


# ============================================================
# 11. CHECK DATASET SIZES
# ============================================================

print("\n========================================")
print("FINAL DATASETS")
print("========================================")

print("Train data:", len(train_data))
print("Val data:", len(val_data))
print("Test data:", len(test_data))








# The old code is here.




# Show length of train , val , test data...
print("Train data size : ",len(train_data))
print("Val data size : ",len(val_data))
print("Test data size : ",len(test_data))
print(train_data[0].x)


# intialize datasets
Nodes_data = pd.DataFrame(data = None, columns = ['power_density_W_mm2' , 'area_mm2' , 'layer_idx' , 'kappa_W_mK' , 'block_type_code' ,'has_tsv' ,'is_hotspot' , 'x_center_mm' , 'y_center_mm' , 'z_center_mm' , 'width_mm' , 'height_mm' , 'exposed_area_mm2'])
Edges_data = pd.DataFrame(data = None , columns = ['thermal_resistance' , 'vertical_lateral' , 'geometric_distance' , 'delta_z' , 'contact_overlap_area' , 'conductivity' , 'thermal_conductance' , 'edge_type' , 'thermal_path_length' , 'material_path' , 'physical_path_info'])
global_data = pd.DataFrame(data = None , columns = [
    'ambient_temp_K',
    'workload_factor',
    'total_power_W',
    'n_layers',
    'n_tsv_pairs',
    'has_hotspot',
    'hotspot_multiplier',
    'cooling_type_code',
    'cooling_htc_w_m2k',
    'package_recipe_code',
    'non_uniform_material',
    'bond_kappa_w_mk',
    'bond_thickness_mm',
    'has_rdl',
    'rdl_thickness_mm',
    'rdl_kappa_w_mk',
    'has_interposer',
    'interposer_thickness_mm',
    'interposer_kappa_w_mk',
    'has_c4',
    'c4_thickness_mm',
    'c4_kappa_w_mk',
    'has_package_substrate',
    'substrate_thickness_mm',
    'substrate_kappa_w_mk'
])


# Buid Nodes Dataset.
for graph in train_data:
    temp_data = pd.DataFrame(data = graph.x , columns = ['power_density_W_mm2' , 'area_mm2' , 
                                                         'layer_idx' , 'kappa_W_mK' , 'block_type_code' 
                                                         ,'has_tsv' ,'is_hotspot' , 'x_center_mm' 
                                                         ,'y_center_mm' , 'z_center_mm' , 'width_mm' , 'height_mm' , 'exposed_area_mm2'])
    Nodes_data = pd.concat([Nodes_data , temp_data] , axis = 0)                           

# Build Edges Dataset.
for graph in train_data:
    temp_data = pd.DataFrame(data = graph.edge_attr , columns = ['thermal_resistance' , 'vertical_lateral' , 'geometric_distance' , 'delta_z' , 'contact_overlap_area' , 'conductivity' , 'thermal_conductance' , 'edge_type' , 'thermal_path_length' , 'material_path' , 'physical_path_info'])
    Edges_data = pd.concat([Edges_data , temp_data] , axis = 0)


# Build the Global features dataset.
for graph in train_data:
    temp_data = pd.DataFrame(data = graph.global_features , columns = [
    'ambient_temp_K',
    'workload_factor',
    'total_power_W',
    'n_layers',
    'n_tsv_pairs',
    'has_hotspot',
    'hotspot_multiplier',
    'cooling_type_code',
    'cooling_htc_w_m2k',
    'package_recipe_code',
    'non_uniform_material',
    'bond_kappa_w_mk',
    'bond_thickness_mm',
    'has_rdl',
    'rdl_thickness_mm',
    'rdl_kappa_w_mk',
    'has_interposer',
    'interposer_thickness_mm',
    'interposer_kappa_w_mk',
    'has_c4',
    'c4_thickness_mm',
    'c4_kappa_w_mk',
    'has_package_substrate',
    'substrate_thickness_mm',
    'substrate_kappa_w_mk',
    'chip_width_mm',
    'chip_height_mm'
])
    global_data = pd.concat([global_data , temp_data] , axis = 0)


# Print Nodes , Edges dataset.
print("Nodes Dataset : ")
print(Nodes_data)
print("Edges Dataset : ")
print(Edges_data)
print("Global features Dataset : ")
print(global_data)



import matplotlib.pyplot as plt
plt.hist(Nodes_data['exposed_area_mm2'])
plt.title(label='exposed_area_mm2')
plt.show()


# Handle temp. (scaller of real_temp)
std_target_temp = StandardScaler()

# Analyze the output y(Temperature).
Train_temp_data = []
for graph in train_data:
    temps = graph.y.detach().cpu().numpy()
    Train_temp_data.extend(temps)
    

Train_temp_data = np.array(Train_temp_data)
Train_temp_data = np.log(Train_temp_data).reshape(-1,1)
std_target_temp.fit(Train_temp_data)





# Power scaller.
std_power_density = StandardScaler()
std_power_density.fit(np.array(Nodes_data['power_density_W_mm2']).reshape(-1,1))

# Area scaller.
std_area = StandardScaler()
std_area.fit(np.array(np.log1p(Nodes_data['area_mm2'])).reshape(-1,1))

# Kappa Scaller.
std_kappa = StandardScaler()
std_kappa.fit(np.array(np.log1p(Nodes_data['kappa_W_mK'])).reshape(-1,1))

# exposed_area_mm2 scaller 
std_exposed_area_mm2 = StandardScaler()
std_exposed_area_mm2.fit(np.array(np.log1p(Nodes_data['exposed_area_mm2'])).reshape(-1,1))

# Helper attributes for one hot encoding
Number_of_layers = int(global_data['n_layers'].max())
Number_of_block_types = int(len(Nodes_data['block_type_code'].value_counts()))


scallers_Node_data = {'std_power_density' : std_power_density, 
                      'std_area' : std_area, 
                      'std_kappa' : std_kappa, 
                      'std_exposed_area_mm2' : std_exposed_area_mm2}

one_hot_Node_data = {'Number_of_layers' : Number_of_layers,
                     'Number_of_block_types' : Number_of_block_types}

# Load scallers and one hot endoing data of Nodes
dump(value= scallers_Node_data , filename = "scallers_Node_data")
dump(value= one_hot_Node_data , filename = "one_hot_Node_data")




def Train_Nodes_data_pipleline(train_data):
    # Start preprocessing loop for Node Features.
    keep = [0 , 1 , 3 , 5 , 6 , 12]
    for graph in train_data:

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

    return train_data   





# Thermal resistance scalling.
std_resistance = StandardScaler()
std_resistance.fit(np.array(np.log1p(Edges_data['thermal_resistance'])).reshape(-1, 1))


# geometric_distance scalling.
std_geometric_distance = StandardScaler()
std_geometric_distance.fit(np.array(Edges_data['geometric_distance']).reshape(-1 , 1))


# Contact area scaller.
std_contact_area = StandardScaler()
std_contact_area.fit(np.array(np.log1p(Edges_data['contact_overlap_area'])).reshape(-1, 1))


# thermal_path_length scaller
std_thermal_path_length = StandardScaler()
std_thermal_path_length.fit(np.array(np.log1p(Edges_data['thermal_path_length'])).reshape(-1, 1))

# Helper values for one hot encoding.
Number_of_edge_types = len(Edges_data['edge_type'].value_counts())
print("Number of edge types : " , Number_of_edge_types)



scallers_Edege_data = {'std_resistance' : std_resistance
                      ,'std_geometric_distance':std_geometric_distance ,
                      'std_contact_area':std_contact_area , 
                      'std_thermal_path_length':std_thermal_path_length}

one_hot_edge_data = {'Number_of_edge_types' : Number_of_edge_types}

dump(value=scallers_Edege_data , filename="scallers_Edege_data")
dump(value= one_hot_edge_data , filename="one_hot_edge_data")




def Train_Edges_data_pipleline(train_data):
    # start preporcessing loop of edge features.
    keep = [0,1,2,3,4,8,9,10]
    for graph in train_data:

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


    return train_data    

        



# Global feature processing.
std_total_power_W = StandardScaler()
std_total_power_W.fit(np.array(np.log1p(global_data['total_power_W'])).reshape(-1 , 1))


#cooling_htc_w_m2k scaller
std_cooling_htc = StandardScaler()
std_cooling_htc.fit(np.array(np.log1p(global_data['cooling_htc_w_m2k'])).reshape(-1 , 1))


# bond_kappa_w_mk scaller
std_bond_kappa_w_mk = StandardScaler()
std_bond_kappa_w_mk.fit(np.array(np.log1p(global_data['bond_kappa_w_mk'])).reshape(-1, 1))


# scaller bond_thickness_mm
std_bond_thickness_mm = StandardScaler()
std_bond_thickness_mm.fit(np.array(global_data['bond_thickness_mm']).reshape(-1, 1))


# scaller ambient_temp_k
std_ambient_temp_K = StandardScaler()
std_ambient_temp_K.fit(np.array(global_data['ambient_temp_K']).reshape(-1, 1))

# scaller workload factor
std_workload_factor = StandardScaler()
std_workload_factor.fit(np.array(global_data['workload_factor']).reshape(-1, 1))

std_n_tsv_pairs = StandardScaler()
std_n_tsv_pairs.fit(np.array(global_data['n_tsv_pairs']).reshape(-1,1))

std_hotspot_multiplier = StandardScaler()
std_hotspot_multiplier.fit(np.array(global_data['hotspot_multiplier']).reshape(-1,1))

std_chip_width_mm = StandardScaler()
std_chip_width_mm.fit(np.array(global_data['chip_width_mm']).reshape(-1,1))


std_chip_height_mm = StandardScaler()
std_chip_height_mm.fit(np.array(global_data['chip_height_mm']).reshape(-1,1))


# helper one hot encoding values
Number_of_cooling_types = len(global_data['cooling_type_code'].value_counts())
Number_of_package_recipe_code = len(global_data['package_recipe_code'].value_counts())





scallers_global_data = {'std_total_power_W' : std_total_power_W , 
                        'std_cooling_htc' : std_cooling_htc , 
                        'std_bond_kappa_w_mk' : std_bond_kappa_w_mk , 
                        'std_bond_thickness_mm' : std_bond_thickness_mm , 
                        'std_ambient_temp_K' : std_ambient_temp_K, 
                        'std_workload_factor' : std_workload_factor , 
                        'std_n_tsv_pairs' : std_n_tsv_pairs , 
                        'std_hotspot_multiplier' : std_hotspot_multiplier , 
                        'std_chip_width_mm' : std_chip_width_mm , 
                        'std_chip_height_mm' : std_chip_height_mm}

one_hot_global_data = {'Number_of_cooling_types' : Number_of_cooling_types , 
                       'Number_of_package_recipe_code' : Number_of_package_recipe_code}

dump(value = scallers_global_data , filename='scallers_global_data')
dump(value = one_hot_global_data , filename='one_hot_global_data')


def Train_Global_data_pipleline(train_data):
    KEEP_GLOBAL = [0, 2, 3, 8, 11, 12]
    for graph in train_data:

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

    return train_data



print("number of package reciepe : ", Number_of_package_recipe_code)
print("number of Number_of_cooling_types : " , Number_of_cooling_types)

def Train_Global_data_pipleline_new(train_data):
    KEEP_GLOBAL = [0, 2, 3 , 5, 8, 11, 12 , 13 , 16 , 19 , 22 ]
    for graph in train_data:

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

    return train_data

def normalize_temp(train_data):
    for graph in train_data:
        col = graph.y.numpy().reshape(-1,1)
        col = np.log(col)
        col = std_target_temp.transform(X=col)
        graph.T_real_normalized = torch.tensor(data = col , dtype = torch.float32).squeeze(dim = 1)

    return train_data


train_data = normalize_temp(train_data= train_data)



normalized_temp_scaller = {"std_target_temp" : std_target_temp} 
dump(value= normalized_temp_scaller , filename="normalized_temp_scaller")