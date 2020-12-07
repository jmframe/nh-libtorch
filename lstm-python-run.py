import numpy as np
import pandas as pd
import torch
from torch import nn
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(0)

class LSTM(nn.Module):
    def __init__(self, input_size=11, hidden_layer_size=64, output_size=1, batch_size=256):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size # In this application we do one timestep only
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_layer, h_t, c_t):
        h_t = h_t.float()
        c_t = c_t.float()
        input_layer = input_layer.float()
        input_view = input_layer.view(batch_size, input_size)
        input_layer = input_view.unsqueeze(1)
        output, (h_t, c_t) = self.lstm(input_layer, (h_t,c_t))
        prediction = self.linear(output)
        return prediction, h_t, c_t

input_size = 11
hidden_layer_size = 64
output_size = 1
batch_size = 336
model = LSTM(input_size, hidden_layer_size, output_size, batch_size)

istart=63578
iend=72338
n = iend - istart

data_dir = './data/'
with open(data_dir+'sugar_creek_input_all.csv','r') as f:
    df = pd.read_csv(f)
input_tensor = torch.tensor(df.iloc[istart:iend,:].values)

with open(data_dir+'sugar_creek_scaler.p', 'rb') as fb:
    scalers = pickle.load(fb)
with  open(data_dir+'sugar_creek_basin_data.csv', 'r') as f:
    sug_crek = pd.read_csv(f)
obs = list(sug_crek.iloc[istart:iend,-1])

pretrained_dict = torch.load(data_dir+'sugar_creek_trained.pt', map_location=torch.device('cpu'))
# Change the name of the "head" layer to linear, since that is what the LSTM expects
pretrained_dict['linear.weight'] = pretrained_dict.pop('head.net.0.weight')
pretrained_dict['linear.bias'] = pretrained_dict.pop('head.net.0.bias')

model.load_state_dict(pretrained_dict)

obs_std = np.array(scalers['xarray_stds'].to_array())[-1]
obs_mean = np.array(scalers['xarray_means'].to_array())[-1]
att_means = np.append(np.array(scalers['attribute_means'].values)[1:3], np.array(scalers['attribute_means'].values)[0])
att_stds = np.append(np.array(scalers['attribute_stds'].values)[1:3], np.array(scalers['attribute_stds'].values)[0])
scaler_mean = np.append(np.array(scalers['xarray_means'].to_array())[:-1], att_means)
scaler_std = np.append(np.array(scalers['xarray_stds'].to_array())[:-1], att_stds)
input_tensor = (input_tensor-scaler_mean)/scaler_std

hidden_layer_size = 64
seq_length = batch_size
h_t = torch.zeros(1, 1, hidden_layer_size).float()
c_t = torch.zeros(1, 1, hidden_layer_size).float()
output_list = []
for t in tqdm(range(seq_length+1,n)):
    with torch.no_grad():
        input_layer = input_tensor[t-seq_length:t, :]
        output, h_t, c_t = model(input_layer, h_t, c_t)
        output = output * obs_std + obs_mean
        output_list.append(output[0,0,0].numpy().tolist())

diff_sum2 = 0
diff_sum_mean2 = 0
obs_mean = np.nanmean(np.array(obs))
print(obs_mean)
for i, j in zip(output_list, obs):
    if np.isnan(j):
        continue
    mod_diff = i-j
    mean_diff = j-obs_mean
    diff_sum2 += np.power((mod_diff),2)
    diff_sum_mean2 += np.power((mean_diff),2)
nse = 1-(diff_sum2/diff_sum_mean2)
print(nse)



