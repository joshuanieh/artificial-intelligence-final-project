import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
class ExampleDataset(Dataset):
    def __init__(self):
        xy = pd.read_csv("./data/keelung.csv", encoding="Big5", skiprows=2, usecols=[i for i in range(3,27)])
        xy = xy.apply(pd.to_numeric, errors='coerce')
        print("origin data: ", xy)
        xy = xy.mean(axis=1, skipna=True, numeric_only=True).values
        xy = xy[None].T
        print("mean: ", xy)
        self.x = np.concatenate((xy[0:-3], xy[1:-2], xy[2:-1]), axis=1)
        self.y = xy[3:]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

dataset = ExampleDataset()

first_data = dataset[0]
features, labels = first_data
print(features, labels)
print(len(dataset))
