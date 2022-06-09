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
          "considered_days": 7,
          "stop_early": 200,
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
        xy = xy.mean(axis=1, skipna=True, numeric_only=True).values
        xy = xy[None].T
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

for region in regions:
  dataset_CO    = AirDataset(1, region, "CO")
  dataset_NO2   = AirDataset(2, region, "NO2")
  dataset_O3    = AirDataset(3, region, "O3")
  dataset_PM10  = AirDataset(4, region, "PM10")
  dataset_PM25  = AirDataset(5, region, "PM25")
  dataset_SO2   = AirDataset(6, region, "SO2")
  datasets = [dataset_CO, dataset_NO2, dataset_O3, dataset_PM10, dataset_PM25, dataset_SO2]
  data_min = [round(data.min, 4) for data in datasets]
  data_max = [round(data.max, 4) for data in datasets]
  print("min: ", data_min)
  print("max: ", data_max)
  # print([len(i) for i in datasets])
  print([i.name for i in datasets])

  predictor_CO   = Predictor(config["considered_days"]).to(device)
  predictor_NO2  = Predictor(config["considered_days"]).to(device)
  predictor_O3   = Predictor(config["considered_days"]).to(device)
  predictor_PM10 = Predictor(config["considered_days"]).to(device)
  predictor_PM25 = Predictor(config["considered_days"]).to(device)
  predictor_SO2  = Predictor(config["considered_days"]).to(device)
  predictors = [predictor_CO, predictor_NO2, predictor_O3, predictor_PM10, predictor_PM25, predictor_SO2]
  test_losses = []
  valid_losses = []
  for gas in range(6):
    print(datasets[gas].name)
    train_set_size = int(config["train_set_ratio"] * len(datasets[gas]))
    valid_set_size = (len(datasets[gas]) - train_set_size)//2
    test_set_size = len(datasets[gas]) - valid_set_size - train_set_size
    test_set, valid_train_set = random_split(datasets[gas], [test_set_size, valid_set_size + train_set_size], generator=torch.Generator().manual_seed(881228)) #Josh's birthday
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(predictors[gas].parameters(), lr=config['learning_rate'], momentum=config["momentum"], weight_decay=config["weight_decay"]) 

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    lowest_loss = 881122 #Jimmy's birthday
    count = 0
    for epoch in range(config["epochs"]):
      valid_set, train_set = random_split(valid_train_set, [valid_set_size, train_set_size], generator=torch.Generator().manual_seed(random.randint(0, 881227)))
      train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
      valid_loader = DataLoader(valid_set, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
      
      # train
      loss_record = []
      predictors[gas].train()
      for x, y in train_loader:
          mask = np.ones(len(y), dtype=bool)
          for i in range(len(mask)):
            if np.isnan(x[i]).any() or np.isnan(y[i]):
              mask[i] = False
          x = x[mask]
          y = y[mask]
          optimizer.zero_grad()
          x, y = x.float().to(device), y.float().to(device)
          # print("x, y: ", x.shape, y.shape)
          pred = predictors[gas](x)
          loss = criterion(pred, y)
          loss.backward()
          optimizer.step()
          loss_record.append(loss.detach().item())
          # print("--------------------------------------")
          # print("INSIDE")
          # print("x: ", x)
          # print("pred: ", pred)
          # print("grad: ", predictors[gas].l1.weight.grad)
          # for param in predictors[gas].parameters():
          #   print("param: ", param.data)
      
      train_loss = sum(loss_record)/len(loss_record)
      
      # validation
      loss_record = []    
      predictors[gas].eval()
      for x, y in valid_loader:
          mask = np.ones(len(y), dtype=bool)
          for i in range(len(mask)):
            if np.isnan(x[i]).any() or np.isnan(y[i]):
              mask[i] = False
          x = x[mask]
          y = y[mask]
          x, y = x.float().to(device), y.float().to(device)
          with torch.no_grad():
              pred = predictors[gas](x) 
              loss = criterion(pred, y)

          loss_record.append(loss.item())
          
      valid_loss = sum(loss_record)/len(loss_record)
      print(f'Epoch {epoch+1}/{config["epochs"]}: Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}')
      if valid_loss < lowest_loss:
          lowest_loss = valid_loss
          torch.save(predictors[gas].state_dict(), f"./models/{datasets[gas].name}_{region}_best.ckpt")
          print("Ya! New model saved.")
          count = 0
      else: 
          count += 1

      if count >= config["stop_early"]:
          print("stop early")
          break
      
    # test
    for x, y in test_loader:
        mask = np.ones(len(y), dtype=bool)
        for i in range(len(mask)):
          if np.isnan(x[i]).any() or np.isnan(y[i]):
            mask[i] = False
        x = x[mask]
        y = y[mask]
        x, y = x.float().to(device), y.float().to(device)
        with torch.no_grad():
            pred = predictors[gas](x) 
            loss = criterion(pred, y)

        loss_record.append(loss.item())
    test_loss = sum(loss_record)/len(loss_record)
    print(f'Test loss: {test_loss:.4f}')
    test_losses.append(round(test_loss, 4))
    valid_losses.append(round(lowest_loss, 4))
  print("test: ", test_losses)
  print("valid: ", valid_losses)