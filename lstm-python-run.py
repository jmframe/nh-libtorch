import numpy as np
import pandas as pd
import torch
from torch import nn
import pickle

torch.manual_seed(0)

class LSTM(nn.Module):
    def __init__(self, input_size=11, hidden_layer_size=100, output_size=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = 1 # In this application we do one timestep only
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_layer, h_t, c_t):
        h_t = h_t.float()
        c_t = c_t.float()
        input_layer = input_layer.float()
        input_view = input_layer.view(1,1,-1)
        output, (c_t, h_t) = self.lstm(input_view, (h_t,c_t))
        prediction = self.linear(h_t)
        return prediction, h_t, c_t

model = LSTM()

print(model)

data_dir = './data/'
with open(data_dir+'sugar_creek_input.csv','r') as f:
    df = pd.read_csv(f)
input_tensor = torch.tensor(df.values)

with open(data_dir+'sugar_creek_scaler.p', 'rb') as fb:
    p = pickle.load(fb)

att_means = np.append(np.array(p['attribute_means'].values)[1:3], np.array(p['attribute_means'].values)[0])
att_stds = np.append(np.array(p['attribute_stds'].values)[1:3], np.array(p['attribute_stds'].values)[0])
scaler_mean = np.append(np.array(p['xarray_means'].to_array())[:-1], att_means)
scaler_std = np.append(np.array(p['xarray_stds'].to_array())[:-1], att_stds)
input_tensor = (input_tensor-scaler_mean)/scaler_std

hidden_layer_size = 100
h_t = torch.zeros(1,1, hidden_layer_size).float()
c_t = torch.zeros(1,1, hidden_layer_size).float()
for t in range(300):
    output, h_t, c_t = model(input_tensor[t,:], h_t, c_t)
    with torch.no_grad():
        print(output * np.array(p['xarray_stds'].to_array())[-1] + np.array(p['xarray_means'].to_array())[-1])






