import matplotlib.pyplot as plt
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
if not os.path.isdir('./models'):
    os.mkdir('./models')

!gdown --id '13Nx3WOTLpjQ1NaXF5ZpVMRN9s533jAxy' --output taiwan.png
def draw(gas_name, guting, banquio, zhongli, xitun, tainan, cianjhen):
  img = plt.imread("taiwan.png")
  fig, ax = plt.subplots()
  ax.imshow(img)
  x = [160, 150, 140, 110, 95, 100]
  y = [40, 43, 45, 90, 140, 170]
  notation = [guting, banquio, zhongli, xitun, tainan, cianjhen]
  # ax.imshow(img, extent=[0, 400, 0, 300])
  ax.scatter(x, y)
  for i, txt in enumerate(notation):
    if i == 0:
      ax.annotate(txt, (x[i], y[i] - 7))
    elif i == 2:
      ax.annotate(txt, (x[i], y[i] + 10))
    else:
      ax.annotate(txt, (x[i] + 15, y[i]))
  ax.set_title(gas_name)
  plt.show()

# !gdown --id '1-fhhnr-3al5qe-cX0aQIyhjeEEHzlx91' --output data/banquio.csv
# !gdown --id '18i3hVfzcNKSvPg1SG-OSauulL9iD98mg' --output data/guting.csv
# !gdown --id '17pgQbZ_6blkwADwbAmOswa6O9hwBLRNY' --output data/zhongli.csv
# !gdown --id '1ZOHBNlTNScftMZVcpyC6XxJuVc8--cvN' --output data/cianjhen.csv
# !gdown --id '15QFzIvi53G4q9oYAR7wSwo_9PxAWvVtJ' --output data/tainan.csv
# !gdown --id '1l_UiWMJhoQIO-IOdduDEF6VA-i-Kt3W1' --output data/xitun.csv

# !gdown --id '1wc9MLT8RHt2gY4lI-g1fM1GtrmGWG2FM' --output models/CO_cianjhen_best.ckpt
# !gdown --id '1sCK35jSXUOWfljk1hTxx6LcGkcLXeVkA' --output models/NO2_cianjhen_best.ckpt
# !gdown --id '1hrc-q1hbUhtOnsLPT9OB20trHutFDmPm' --output models/O3_cianjhen_best.ckpt
# !gdown --id '1HphoN4ZDAXfZwBA8y2i7H-v9iZJSdRyB' --output models/PM10_cianjhen_best.ckpt
# !gdown --id '1eg8MmqMjNDuPcP5TG9pkxzEbKVPSF-7C' --output models/PM25_cianjhen_best.ckpt
# !gdown --id '13tDozTuh39jvEOAyXEevuWMCzVh1_LrM' --output models/SO2_cianjhen_best.ckpt

# !gdown --id '10KSTxDNIISt3sBzHPeQEMmk_yRG7A6K-' --output models/CO_tainan_best.ckpt
# !gdown --id '1KqW2Ey_cP60NZdnb28AFb6TwZP400QgV' --output models/NO2_tainan_best.ckpt
# !gdown --id '1K1dG66P34imxNon_0dpqqw3-sHhisiFl' --output models/O3_tainan_best.ckpt
# !gdown --id '11egDR2yNshFzApGB6kjxMy1SroydnVIe' --output models/PM10_tainan_best.ckpt
# !gdown --id '1lWlTgYJrvqG5-M8IxGYEYxVX49eiA1g1' --output models/PM25_tainan_best.ckpt
# !gdown --id '1IvuXH2KEbdVKo6FzrvrOLYoiwezHBmEi' --output models/SO2_tainan_best.ckpt

# !gdown --id '10jAf5OddnHwya2VJLDEqPSvLtQI-5tG1' --output models/CO_xitun_best.ckpt
# !gdown --id '1arBPCk4Kt9NYUAxY_-bOnqyRKnWUzafA' --output models/NO2_xitun_best.ckpt
# !gdown --id '1jvpR-31PtMD1q6VHlIu-n5SM8laMwPIM' --output models/O3_xitun_best.ckpt
# !gdown --id '1Xf-LTqW2gyjB7GPY4wM9GJVeQVCaZJZk' --output models/PM10_xitun_best.ckpt
# !gdown --id '1oAKOMr7swl0lXy02Su7MpVzySCT19A-i' --output models/PM25_xitun_best.ckpt
# !gdown --id '1VWZ6m4Hk1RUvzielfZiU6nSxfZMF7HWj' --output models/SO2_xitun_best.ckpt

# !gdown --id '1Joqgj53MJQNLccpTZ9GcqdQuSjDwear3' --output models/CO_banquio_best.ckpt
# !gdown --id '1Z6HJ3l5AV1w7AxucaHqlo0rkaLsJmf2-' --output models/NO2_banquio_best.ckpt
# !gdown --id '1TOwJf3A9yq5S1PW-PLTSu-eKWV15oiwj' --output models/O3_banquio_best.ckpt
# !gdown --id '19o0nnEJPUw-VUZnCn4Q3_Sp21vha-Ijy' --output models/PM10_banquio_best.ckpt
# !gdown --id '1LBWuwTe5NDmfbJr4HgP03yaTJUp0W-qk' --output models/PM25_banquio_best.ckpt
# !gdown --id '1FrU5hm729WzSUJ5q3ddG-i26d8IH4yLW' --output models/SO2_banquio_best.ckpt

# !gdown --id '1H0uPhFWgowNQCkeIybzMKNruOwyrEuKw' --output models/CO_guting_best.ckpt
# !gdown --id '1Bfhbf2jIT6nJVgG2VhK-MJR6haoX_m08' --output models/NO2_guting_best.ckpt
# !gdown --id '1RA2LFfw4KBR61Rk7uj7-xijYbkPBDUdK' --output models/O3_guting_best.ckpt
# !gdown --id '1w2sKMYXLLADA0FdBgr9rPgXnYTEBkUPI' --output models/PM10_guting_best.ckpt
# !gdown --id '19D0ZelaRhGs4GcngEshzda1ta3-3eojs' --output models/PM25_guting_best.ckpt
# !gdown --id '1Z2c5yq1d-lwmiyWwg5mxRZZDlop6D38G' --output models/SO2_guting_best.ckpt

# !gdown --id '1FwT0nokpso_3ngSGnezEPXVzoWELZ4VN' --output models/CO_zhongli_best.ckpt
# !gdown --id '1S1veoqDodbPkmiGRmd9TJqu83TTT8QMP' --output models/NO2_zhongli_best.ckpt
# !gdown --id '1Si1zhi7g6oOw9lfuQ-MhTasXad7ozgHA' --output models/O3_zhongli_best.ckpt
# !gdown --id '1qHRE2aLSi3KDqOuS2877s7uSi9BvstLq' --output models/PM10_zhongli_best.ckpt
# !gdown --id '1y7vtCT_AI7cp9RCOHCVPQj7BA4w549gP' --output models/PM25_zhongli_best.ckpt
# !gdown --id '1Ax_hPFDPfmV61DPN-Lc_khZx5779-5tL' --output models/SO2_zhongli_best.ckpt

config = {"train_set_ratio": 0.6, 
          "epochs": 1000,
          "batch_size": 60,
          "considered_days": 7,
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
# print(data.shape[0])
gas = ["CO", "NO2", "O3", "PM10", "PM25", "SO2"]
regions = ["guting", "banquio", "zhongli", "xitun", "tainan", "cianjhen"]
record = dict([(i, []) for i in gas])
for region in range(6):
  # print(regions[region])
  for i in range(6):
    # print(gas[i])
    saved_model = Predictor(config["considered_days"])
    saved_model.load_state_dict(torch.load(f'./models/{gas[i]}_{regions[i]}_best.ckpt'))
    saved_model.eval()
    #print(datasets[i].mean[data.shape[0]//6*(i+1)-7-j:data.shape[0]//6*(i+1)-j])
    for j in range(0, -1, -1):
      mean = datasets[region][i].mean[data.shape[0]//6*(i+1)-config["considered_days"]-j:data.shape[0]//6*(i+1)-j]
      mean = (mean - datasets[region][i].min)/(datasets[region][i].max - datasets[region][i].min)
      try:
      # print(saved_model(torch.from_numpy(datasets[region][i].mean[data.shape[0]//6*(i+1)-7-j:data.shape[0]//6*(i+1)-j]).float()).item())
          # print((saved_model(torch.from_numpy(mean).float())*(datasets[region][i].max - datasets[region][i].min) + datasets[region][i].min).item())
          record[gas[i]].append((saved_model(torch.from_numpy(mean).float())*(datasets[region][i].max - datasets[region][i].min) + datasets[region][i].min).item())
      except:
          record[gas[i]].append(float("nan"))

for i in gas:
  # print(record[i])
  draw(i, *record[i])
