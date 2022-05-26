import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import os

config = {"train_set_ratio": 0.6, 
          "epochs": 10,
          "batch_size": 60,
          "considered days": 7,
          "stop_early": 200,
          "learning_rate": 0.01,
          "momentum": 0.9,
          "model_path": "./models/best.ckpt"}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class Predictor(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(i, 16),
          nn.ReLU(),
          nn.Linear(16, 8),
          nn.ReLU(),
          nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x

dataset_CO = AirDataset(1)
predictor_CO = Predictor(config["considered days"])
# first_data = dataset_CO[0]
# features, labels = first_data
# print(features, labels)
# print(len(dataset_CO))

train_set_size = int(config["train_set_ratio"] * len(dataset_CO))
valid_set_size = (len(dataset_CO) - train_set_size)//2
test_set_size = len(dataset_CO) - valid_set_size - train_set_size
print(len(dataset_CO), train_set_size, valid_set_size, test_set_size)
test_set, valid_train_set = random_split(dataset_CO, [test_set_size, valid_set_size + train_set_size], generator=torch.Generator().manual_seed(881228)) #Josh's birthday
test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=3, pin_memory=True)

criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(predictor_CO.parameters(), lr=config['learning_rate'], momentum=config['momentum']) 

if not os.path.isdir('./models'):
    os.mkdir('./models')

lowest_loss = 881122 #Jimmy's birthday
count = 0
for epoch in range(config["epochs"]):
    valid_set, train_set = random_split(valid_train_set, [valid_set_size, train_set_size], generator=torch.Generator().manual_seed(random.randint(0, 881227)))
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=3, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=config["batch_size"], shuffle=True, num_workers=3, pin_memory=True)
    
    loss_record = []
    predictor_CO.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.float().to(device), y.float().to(device)
        pred = predictor_CO(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        loss_record.append(loss.detach().item())
    
    train_loss = sum(loss_record)/len(loss_record)
    
    loss_record = []    
    predictor_CO.eval()
    for x, y in valid_loader:
        x, y = x.float().to(device), y.float().to(device)
        with torch.no_grad():
            pred = predictor_CO(x)
            loss = criterion(pred, y)

        loss_record.append(loss.item())
        
    valid_loss = sum(loss_record)/len(loss_record)
    print(f'Epoch {epoch+1}/{config["epochs"]}: Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}')
    if valid_loss < lowest_loss:
        lowest_loss = valid_loss
        torch.save(predictor_CO.state_dict(), config['model_path'])
        print("Ya! New model saved.")
        count = 0
    else: 
        count += 1

    if count >= config["stop_early"]:
        print("stop early")
        break
