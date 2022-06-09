data = pd.read_csv("./data/zhongli.csv", encoding="Big5", skiprows=2, usecols=range(3,27))
print(data.shape[0])
models = ["CO", "NO2", "O3", "PM10", "PM25", "SO2"]
for i in range(6):
  print(models[i])
  Josh = Predictor(7).to(device)
  Josh.load_state_dict(torch.load("./models/" + models[i] + "_best.ckpt"))
  Josh.eval()
  #print(datasets[i].mean[data.shape[0]//6*(i+1)-7-j:data.shape[0]//6*(i+1)-j])
  for j in range(6, -1, -1):
    print((Josh(torch.from_numpy(datasets[i].mean[data.shape[0]//6*(i+1)-7-j:data.shape[0]//6*(i+1)-j]).float())*(datasets[i].max - datasets[i].min) + datasets[i].min).item())