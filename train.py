import torch
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
import pandas as pd

import random
config = {"train_set_ratio": 0.6, 
          "epochs": 10,
          "batch_size": 60,
          "considered days": 3}

class AirDataset(Dataset):
    def __init__(self, i):
        #i = 1~6
        xy = pd.read_csv("./data/keelung.csv", encoding="Big5", skiprows=2, usecols=[i for i in range(3,27)])
        xy = xy.apply(pd.to_numeric, errors='coerce')
        print("origin data: ", xy)
        xy = xy.mean(axis=1, skipna=True, numeric_only=True).values
        xy = xy[None].T
        print("mean: ", xy)
        self.x = np.concatenate([xy[len(xy)*(i-1)//6+j:len(xy)*i//6-config["considered days"]+j] for j in range(config["considered days"])], axis=1)
        print("x: ", self.x)
        self.y = xy[len(xy)*(i-1)//6+3:len(xy)*i//6]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

# class Predictor(nn.Module):
#     def __init__(self, input_dim):
#         super(My_Model, self).__init__()
#         # TODO: modify model's structure, be aware of dimensions. 
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, 16),
#             nn.ReLU(),
#             nn.Linear(16, 8),
#             nn.ReLU(),
#             nn.Linear(8, 1)
#         )

#     def forward(self, x):
#         x = self.layers(x)
#         x = x.squeeze(1) # (B, 1) -> (B)
#         return x

dataset_CO = AirDataset(0)

first_data = dataset_CO[0]
features, labels = first_data
print(features, labels)
print(len(dataset_CO))

train_set_size = int(config["train_set_ratio"] * len(dataset_CO))
valid_set_size = (len(dataset_CO) - train_set_size)//2
test_set_size = len(dataset_CO) - valid_set_size - train_set_size
print(len(dataset_CO), train_set_size, valid_set_size, test_set_size)
test_set, valid_train_set = random_split(dataset_CO, [test_set_size, valid_set_size + train_set_size], generator=torch.Generator().manual_seed(881228))
test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=3, pin_memory=True)


for epoch in range(config["epochs"]):
  valid_set, train_set = random_split(valid_train_set, [valid_set_size, train_set_size], generator=torch.Generator().manual_seed(random.randint(0, 881227)))
  train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=3, pin_memory=True)
  valid_loader = DataLoader(valid_set, batch_size=config["batch_size"], shuffle=True, num_workers=3, pin_memory=True)
