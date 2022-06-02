import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import os

if not os.path.isdir('./data'):
    os.mkdir('./data')
!gdown --id '1kXEhAtn6JfpYtTsQ9BkJVxKxf47ASODk' --output data/keelung.csv
        
config = {"train_set_ratio": 0.6, 
          "epochs": 1000,
          "batch_size": 60,
          "considered_days": 3,
          "stop_early": 200,
          "learning_rate": 0.001,
          "momentum": 0.9}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AirDataset(Dataset):
    def __init__(self, i, name):
        #i = 1~6
        xy = pd.read_csv("./data/keelung.csv", encoding="Big5", skiprows=2, usecols=range(3,27))
        xy = xy.apply(pd.to_numeric, errors='coerce')
        # print("origin data: ", xy)
        xy = xy.mean(axis=1, skipna=True, numeric_only=True).values
        xy = xy[None].T
        # print("mean: ", xy)
        self.x = np.concatenate([xy[len(xy)*(i-1)//6+j:len(xy)*i//6-config["considered_days"]+j] for j in range(config["considered_days"])], axis=1)
        # print(name)
        # print("x: ", self.x)
        self.y = xy[len(xy)*(i-1)//6+config["considered_days"]:len(xy)*i//6]
        self.name = name

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class Predictor(nn.Module):
    def __init__(self, i):
        super().__init__()
        # self.layers = nn.Sequential(
        #   nn.Linear(i, 16),
        #   nn.ReLU(),
        #   nn.Linear(16, 8),
        #   nn.ReLU(),
        #   nn.Linear(8, 1)
        # )
        self.l1  = nn.Linear(i, 16)
        self.r1  = nn.ReLU()
        self.l2 = nn.Linear(16, 8)
        self.r2 = nn.ReLU()
        self.l3 = nn.Linear(8, 1)

    def forward(self, x):
        # print("x: ", x.shape)
        # print("x: ", x)
        # x = self.layers(x)
        x = self.l1(x)
        # print("x1: ", x.shape)
        # print("x1: ", x)
        x = self.r1(x)
        x = self.l2(x)
        x = self.r2(x)
        x = self.l3(x)
        # print("x:", x)
        # print(x.shape)
        # x = x.squeeze(1)
        # print(x.shape)
        return x

dataset_CO    = AirDataset(1, "CO")
dataset_NO2   = AirDataset(2, "NO2")
dataset_O3    = AirDataset(3, "O3")
dataset_PM10  = AirDataset(4, "PM10")
dataset_PM25  = AirDataset(5, "PM25")
dataset_SO2   = AirDataset(6, "SO2")
datasets = [dataset_CO, dataset_NO2, dataset_O3, dataset_PM10, dataset_PM25, dataset_SO2]
# print([len(i) for i in datasets])
print([i.name for i in datasets])

predictor_CO   = Predictor(config["considered_days"])
predictor_NO2  = Predictor(config["considered_days"])
predictor_O3   = Predictor(config["considered_days"])
predictor_PM10 = Predictor(config["considered_days"])
predictor_PM25 = Predictor(config["considered_days"])
predictor_SO2  = Predictor(config["considered_days"])
predictors = [predictor_CO, predictor_NO2, predictor_O3, predictor_PM10, predictor_PM25, predictor_SO2]

for gas in range(0,6):
  print(datasets[gas].name)
  train_set_size = int(config["train_set_ratio"] * len(datasets[gas]))
  valid_set_size = (len(datasets[gas]) - train_set_size)//2
  test_set_size = len(datasets[gas]) - valid_set_size - train_set_size
  test_set, valid_train_set = random_split(datasets[gas], [test_set_size, valid_set_size + train_set_size], generator=torch.Generator().manual_seed(881228)) #Josh's birthday
  test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
  criterion = nn.MSELoss(reduction='mean')
  optimizer = torch.optim.SGD(predictors[gas].parameters(), lr=config['learning_rate'], momentum=config['momentum']) 

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
        torch.save(predictors[gas].state_dict(), f"./models/{datasets[gas].name}_best.ckpt")
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