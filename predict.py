import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt

if not os.path.isdir('./data'):
    os.mkdir('./data')
    
# !gdown --id '1-fhhnr-3al5qe-cX0aQIyhjeEEHzlx91' --output data/banquio.csv
# !gdown --id '18i3hVfzcNKSvPg1SG-OSauulL9iD98mg' --output data/guting.csv
# !gdown --id '17pgQbZ_6blkwADwbAmOswa6O9hwBLRNY' --output data/zhongli.csv
# !gdown --id '1ZOHBNlTNScftMZVcpyC6XxJuVc8--cvN' --output data/cianjhen.csv
# !gdown --id '15QFzIvi53G4q9oYAR7wSwo_9PxAWvVtJ' --output data/tainan.csv
# !gdown --id '1l_UiWMJhoQIO-IOdduDEF6VA-i-Kt3W1' --output data/xitun.csv

config = {"train_set_ratio": 0.6, 
          "epochs": 1000,
          "batch_size": 60,
          "considered_days": 28,
          "stop_early": 100,
          "learning_rate": 0.0001,
          "momentum": 0.9, 
          "weight_decay": 0.000001,
          "init": False}
regions = ["guting", "banquio", "zhongli", "xitun", "tainan", "cianjhen"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AirDataset(Dataset):
    def __init__(self, i, region, name):
        #i = 1~6
        xy = pd.read_csv(f'./data/{region}.csv', encoding="Big5", skiprows=2, usecols=range(3,27))
        xy = xy.apply(pd.to_numeric, errors='coerce')
        # print("origin data: ", xy)
        self.mean = xy.mean(axis=1, skipna=True, numeric_only=True).values
        xy = self.mean[None].T
        # print("mean: ", xy)
        self.x = np.concatenate([xy[len(xy)*(i-1)//6+j:len(xy)*i//6-config["considered_days"]+j] for j in range(config["considered_days"])], axis=1)
        # print(name)
        # print("x: ", self.x)
        self.y = xy[len(xy)*(i-1)//6+config["considered_days"]:len(xy)*i//6]
        self.min, self.max = min(np.nanmin(self.x), np.nanmin(self.y)), max(np.nanmax(self.x), np.nanmax(self.y))
        self.x = (self.x-self.min)/(self.max-self.min)
        self.y = (self.y-self.min)/(self.max-self.min)
        self.name = name

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class Predictor(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Dropout(0.5),
          nn.Linear(i, 16),
          nn.ReLU(),
          nn.Linear(16, 8),
          nn.ReLU(),
          nn.Linear(8, 1)
        )
        if config["init"]:
          self.initialize_weights()

    def forward(self, x):
        # print("x: ", x.shape)
        # print("x: ", x)
        x = self.layers(x)
        # print("x1: ", x.shape)
        # print("x1: ", x)
        # print("x:", x)
        # print(x.shape)
        # x = x.squeeze(1)
        # print(x.shape)
        return x
    
    def initialize_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Conv1d):
          torch.nn.init.xavier_normal_(m.weight.data)
          if m.bias is not None:
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
          m.weight.data.fill_(1)
          m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
          torch.nn.init.normal_(m.weight.data, 0, 0.01)
          # m.weight.data.normal_(0,0.01)
          m.bias.data.zero_()

datasets = []
for region in regions:
  dataset_CO    = AirDataset(1, region, "CO")
  dataset_NO2   = AirDataset(2, region, "NO2")
  dataset_O3    = AirDataset(3, region, "O3")
  dataset_PM10  = AirDataset(4, region, "PM10")
  dataset_PM25  = AirDataset(5, region, "PM25")
  dataset_SO2   = AirDataset(6, region, "SO2")
  datasets.append([dataset_CO, dataset_NO2, dataset_O3, dataset_PM10, dataset_PM25, dataset_SO2])

  # print([len(i) for i in datasets])
  # print([(region, i.name) for i in datasets])
data = pd.read_csv("./data/guting.csv", encoding="Big5", skiprows=2, usecols=range(3,27))
print(data.shape[0])
gas = ["CO", "NO2", "O3", "PM10", "PM25", "SO2"]
regions = ["guting", "banquio", "zhongli", "xitun", "tainan", "cianjhen"]
for region in range(6):
  print(regions[region])
  for i in range(6):
    print(gas[i])
    saved_model = Predictor(config["considered_days"])
    saved_model.load_state_dict(torch.load(f'./models/{gas[i]}_{region}_best.ckpt'))
    saved_model.eval()
    #print(datasets[i].mean[data.shape[0]//6*(i+1)-7-j:data.shape[0]//6*(i+1)-j])
    for j in range(6, -1, -1):
      mean = datasets[region][i].mean[data.shape[0]//6*(i+1)-config["considered_days"]-j:data.shape[0]//6*(i+1)-j]
      mean = (mean - datasets[region][i].min)/(datasets[region][i].max - datasets[region][i].min)
      try:
      # print(saved_model(torch.from_numpy(datasets[region][i].mean[data.shape[0]//6*(i+1)-7-j:data.shape[0]//6*(i+1)-j]).float()).item())
          print((saved_model(torch.from_numpy(mean).float())*(datasets[region][i].max - datasets[region][i].min) + datasets[region][i].min).item())
      except:
          pass
