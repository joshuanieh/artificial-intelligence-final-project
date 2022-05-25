import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
class ExampleDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./dataset-example.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        
        return self.x[index], self.y[index]

    def __len__(self):
           
        return self.n_samples