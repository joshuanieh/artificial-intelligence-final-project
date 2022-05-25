import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import pandas as pd

import random
class AirDataset(Dataset):
    def __init__(self, i):
        #i = 1~6
        xy = pd.read_csv("./data/keelung.csv", encoding="Big5", skiprows=2, usecols=[i for i in range(3,27)])
        xy = xy.apply(pd.to_numeric, errors='coerce')
        print("origin data: ", xy)
        xy = xy.mean(axis=1, skipna=True, numeric_only=True).values
        xy = xy[None].T
        print("mean: ", xy)
        self.x = np.concatenate((xy[len(xy)*(i-1)//6:len(xy)*i//6-3], xy[len(xy)*(i-1)//6+1:len(xy)*i//6-2], xy[len(xy)*(i-1)//6+2:len(xy)*i//6-1]), axis=1)
        self.y = xy[len(xy)*(i-1)//6+3:len(xy)*i//6]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

dataset_CO = AirDataset(0)

# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)
print(len(dataset_CO))

train_set_size = int(0.6 * len(dataset_CO))
valid_set_size = (len(dataset_CO) - train_set_size)//2
test_set_size = len(dataset_CO) - valid_set_size - train_set_size
test_set, valid_set = random_split(dataset_CO, [test_set_size, valid_set_size + train_set_size], generator=torch.Generator().manual_seed(881228))
valid_set, train_set = random_split(valid_set, [valid_set_size, train_set_size], generator=torch.Generator().manual_seed(random.randint(0, 881227)))