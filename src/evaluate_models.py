import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from src.loader import loader
import os
print("Using cuda? ", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

_, dataset_name, split, model = sys.argv

print(dataset_name, model)

net, data, dim, channel = loader(model=model, split=split)
# data = torch.utils.data.Subset(data, np.arange(200))
dataset_len = data.__len__()
batch_size = 128
dataloader = DataLoader(data, batch_size=batch_size)

correct_indices = []

for i, (data, labels) in enumerate(dataloader):
    data = data.to(device)
    labels = labels.to(device)
    preds = net(data).argmax(1) 
    
    # Find the indices of correctly classified samples
    indices = list(torch.where(preds == labels.squeeze())[0].cpu())
    indices = [j+i*batch_size for j in indices]
    correct_indices += indices

print("Accuracy: ", len(correct_indices)/dataset_len)

path = f"data/correct_indices/{dataset_name}"
if not os.path.exists(path): os.makedirs(path)

np.save(path + f"/{model}_{split}.npy", np.array(correct_indices))