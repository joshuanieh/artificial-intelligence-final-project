import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
class ExampleDataset(Dataset):
    def __init__(self):
        xy = pd.read_csv("./data/keelung.csv", encoding="Big5", skiprows=2, usecols=[i for i in range(3,27)])
        xy = xy.apply(pd.to_numeric, errors='coerce')
        print("origin data: ", xy)
        xy = xy.mean(axis=1, skipna=True, numeric_only=True)
        print("mean: ", xy)
        #print(xy[1:-2].T)
        a = xy[1:-2].reset_index(drop=True)
        b = xy[2:-1].reset_index(drop=True)
        self.x = pd.concat([xy[0:-3].T, a.T, b.T], axis=1)
        print(self.x)
        self.y = xy[3:]

    def __getitem__(self, index):        
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

dataset = ExampleDataset()

'''first_data = dataset[0]
features, labels = first_data
print(features, labels)
print(len(dataset))'''
