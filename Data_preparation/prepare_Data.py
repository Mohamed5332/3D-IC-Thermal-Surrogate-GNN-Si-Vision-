import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import torch
from torch_geometric.loader import DataLoader



from Data_preparation.Dataset import train_data , test_data , val_data

torch.save(train_data, "D:/Thermal surrogate project/train_raw_graphs.pt")
torch.save(val_data, "D:/Thermal surrogate project/val_raw_graphs.pt")
torch.save(test_data, "D:/Thermal surrogate project/test_raw_graphs.pt")

print(f"Saved Train: {len(train_data)} graphs")
print(f"Saved Val: {len(val_data)} graphs")
print(f"Saved Test: {len(test_data)} graphs")



from Preprocessing.Train_data_processin import Train_Global_data_pipleline_new , Train_Global_data_pipleline  , Train_Nodes_data_pipleline , Train_Edges_data_pipleline , Train_normalize_temp
from Preprocessing.Val_data_processing import val_Global_data_pipleline_new , Val_Global_data_pipleline , Val_Edges_data_pipleline , Val_Nodes_data_pipleline , val_normalize_temp
from Preprocessing.Test_data_processing import test_Global_data_pipleline_new, test_Global_data_pipleline , test_Nodes_data_pipleline , test_Edges_data_pipleline , test_normalize_temp
from sklearn.model_selection import train_test_split
from joblib import dump , load
import torch
from collections import Counter




train_data = Train_Nodes_data_pipleline(train_data = train_data)


print("NODE FEATURE DIM:", train_data[0].x.shape)
print("NODE FEATURES FIRST GRAPH:")
print(train_data[0].x[0])


train_data = Train_Edges_data_pipleline(train_data = train_data)
train_data = Train_Global_data_pipleline_new(train_data = train_data) # edit here..
train_data = Train_normalize_temp(train_data = train_data)


val_data = Val_Nodes_data_pipleline(Val_data = val_data)
val_data = Val_Edges_data_pipleline(Val_data = val_data)
val_data = val_Global_data_pipleline_new(val_data = val_data) # edit here..
val_data = val_normalize_temp(val_data= val_data)

test_data = test_Nodes_data_pipleline(test_data = test_data)
test_data = test_Edges_data_pipleline(test_data = test_data)
test_data = test_Global_data_pipleline_new(test_data = test_data) # edit here..
test_data = test_normalize_temp(test_data = test_data)



def analyze_temperature_distribution(dataset):

    ranges = [
        (300, 350),
        (350, 400),
        (400, 450),
        (450, 500),
    ]

    counts = Counter()

    for graph in dataset:

        # استخدم real_y لو موجود
        temperature = graph.y

        temperature = temperature.detach().cpu()

        for low, high in ranges:

            mask = (
                (temperature >= low) &
                (temperature < high)
            )

            counts[(low, high)] += mask.sum().item()

    print("\n===== TEMPERATURE DISTRIBUTION =====")

    total = sum(counts.values())

    for low, high in ranges:

        count = counts[(low, high)]

        percentage = (
            100.0 * count / total
            if total > 0 else 0
        )

        print(
            f"{low}-{high} K | "
            f"Nodes: {count} | "
            f"Percentage: {percentage:.2f}%"
        )

    print(f"\nTotal nodes: {total}")




# New Test Takes place here...
analyze_temperature_distribution(train_data)
analyze_temperature_distribution(val_data)


loaders = {'train' : DataLoader(dataset= train_data , batch_size= 8 , shuffle= True) , 
           'val' : DataLoader(dataset= val_data , batch_size= 8 , shuffle= True), 
           'test' : DataLoader(dataset= test_data , batch_size= 8 , shuffle= True)}


dump(value = loaders , filename="loaders")

print("\n========== LOADER VERIFICATION ==========")

train_batch = next(iter(loaders["train"]))

print("Node features     :", train_batch.x.shape)
print("Edge features     :", train_batch.edge_attr.shape)
print("Global features   :", train_batch.global_features.shape)

if hasattr(train_batch, "T_real_normalized"):
    print("Normalized target :", train_batch.T_real_normalized.shape)

print("Target y          :", train_batch.y.shape)

print("=========================================")







import numpy as np

def analyze_target(data, name):

    values = []

    for graph in data:
        values.extend(
            graph.y.detach().cpu().numpy().reshape(-1)
        )

    values = np.array(values)

    print(f"\n========== {name} ==========")
    print(f"Count  : {len(values)}")
    print(f"Min    : {values.min():.3f}")
    print(f"Max    : {values.max():.3f}")
    print(f"Mean   : {values.mean():.3f}")
    print(f"Std    : {values.std():.3f}")
    print(f"Median : {np.median(values):.3f}")

    p = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])

    print(f"P1     : {p[0]:.3f}")
    print(f"P5     : {p[1]:.3f}")
    print(f"P25    : {p[2]:.3f}")
    print(f"P50    : {p[3]:.3f}")
    print(f"P75    : {p[4]:.3f}")
    print(f"P95    : {p[5]:.3f}")
    print(f"P99    : {p[6]:.3f}")




analyze_target(train_data, "TRAIN")
analyze_target(val_data, "VALIDATION")