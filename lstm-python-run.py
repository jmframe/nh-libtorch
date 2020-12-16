import numpy as np
import pandas as pd
import torch
from torch import nn
import pickle
import matplotlib as plt

torch.manual_seed(0)

class xLSTM(nn.Module):
    def __init__(self, input_size=11, hidden_layer_size=64, output_size=1, batch_size=1, seq_length=1):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.seq_length=seq_length
        self.batch_size = batch_size # We shouldn't neeed to do a higher batch size.
        self.lstm = nn.LSTM(input_size, hidden_layer_size)

    def forward(self, input_layer, h_t, c_t):
        h_t = h_t.float()
        c_t = c_t.float()
        input_layer = input_layer.float()
        input_view = input_layer.view(seq_length, batch_size, input_size)
        output, (h_t, c_t) = self.lstm(input_view, (h_t,c_t))
        return output, (h_t, c_t)

class xHEAD(nn.Module):
    def __init__(self, hidden_layer_size=64, output_size=1):
        super(xHEAD, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.head = nn.Linear(hidden_layer_size, output_size)

    def forward(self, hidden_layer):
        prediction = self.head(h_t)
        return prediction

input_size = 11
hidden_layer_size = 64
output_size = 1
seq_length = 1
batch_size = 1
hidden_layer_size = 64
model = xLSTM(input_size, hidden_layer_size, output_size, batch_size, seq_length)
head = xHEAD(hidden_layer_size, output_size)

istart=71593    # 71593
iend=72338
warmup = np.maximum(seq_length, 336)
do_warmup = True

data_dir = './data/'
with open(data_dir+'sugar_creek_input_all2.csv','r') as f:
    df = pd.read_csv(f)
df = df.drop(['date','obs'], axis=1)
df = df.loc[:,['RAINRATE', 'Q2D', 'T2D', 'LWDOWN',  'SWDOWN',  'PSFC',  'U2D', 'V2D', 'area_sqkm', 'lat', 'lon']]
input_tensor = torch.tensor(df.values)
print(input_tensor[istart-warmup,:])

with open(data_dir+'sugar_creek_scaler.p', 'rb') as fb:
    scalers = pickle.load(fb)
with  open(data_dir+'sugar_creek_basin_data.csv', 'r') as f:
    sug_crek = pd.read_csv(f)
obs = list(sug_crek.iloc[istart:iend,-1])

p_dict = torch.load(data_dir+'sugar_creek_trained.pt', map_location=torch.device('cpu'))
m_dict = model.state_dict()
lstm_weights = {x:p_dict[x] for x in m_dict.keys()}
head_weights = {}
head_weights['head.weight'] = p_dict.pop('head.net.0.weight')
head_weights['head.bias'] = p_dict.pop('head.net.0.bias')
model.load_state_dict(lstm_weights)
head.load_state_dict(head_weights)

obs_std = np.array(scalers['xarray_stds'].to_array())[-1]
obs_mean = np.array(scalers['xarray_means'].to_array())[-1]
xm = np.array(scalers['xarray_means'].to_array())[:-1]
xs = np.array(scalers['xarray_stds'].to_array())[:-1]
xm2 = np.array([xm[x] for x in list([3,2,5,0,4,1,6,7])])
xs2 = np.array([xs[x] for x in list([3,2,5,0,4,1,6,7])])
att_means = np.append( np.array(scalers['attribute_means'].values)[0],np.array(scalers['attribute_means'].values)[1:3])
att_stds = np.append(np.array(scalers['attribute_stds'].values)[0], np.array(scalers['attribute_stds'].values)[1:3])
scaler_mean = np.append(xm2, att_means)
scaler_std = np.append(xs2, att_stds)
input_tensor = (input_tensor-scaler_mean)/scaler_std
print(input_tensor[istart-warmup,:])

if do_warmup:
    h_t = torch.zeros(1, batch_size, hidden_layer_size).float()
    c_t = torch.zeros(1, batch_size, hidden_layer_size).float()
    for t in range(istart-warmup, istart):
        with torch.no_grad():
            input_layer = input_tensor[t-seq_length:t, :]
            output, (h_t, c_t) = model(input_layer, h_t, c_t)
            h_t = h_t.transpose(0,1)
            c_t = c_t.transpose(0,1)
            if t == istart-1:
                h_t_np = h_t[0,0,:].numpy()
                h_t_df = pd.DataFrame(h_t_np)
                h_t_df.to_csv('data/h_t_start.csv')
                c_t_np = c_t[0,0,:].numpy()
                c_t_df = pd.DataFrame(c_t_np)
                c_t_df.to_csv('data/c_t_start.csv')

h_t = np.genfromtxt('data/h_t_start.csv', skip_header=1, delimiter=",")[:,1]
h_t = torch.tensor(h_t).view(1,1,-1)
c_t = np.genfromtxt('data/c_t_start.csv', skip_header=1, delimiter=",")[:,1]
c_t = torch.tensor(c_t).view(1,1,-1)
output_list = []
for t in range(istart, iend):
    with torch.no_grad():
        input_layer = input_tensor[t-seq_length:t, :]
        lstm_output, (h_t, c_t) = model(input_layer, h_t, c_t)
        h_t = h_t.transpose(0,1)
        c_t = c_t.transpose(0,1)
        output = head(lstm_output.transpose(0,1))
        output = output[0,0,0].numpy().tolist() * obs_std + obs_mean
        output_list.append(output)

print('output stats')
print('mean', np.mean(output_list))
print('min', np.min(output_list))
print('max', np.max(output_list))
print('observation stats')
print('mean', np.nanmean(obs))
print('min', np.nanmin(obs))
print('max', np.nanmax(obs))

diff_sum2 = 0
diff_sum_mean2 = 0
obs_mean = np.nanmean(np.array(obs))
count_samples = 0
for j, k in zip(output_list, obs):
    if np.isnan(k):
        continue
    count_samples += 1
    mod_diff = j-k
    mean_diff = k-obs_mean
    diff_sum2 += np.power((mod_diff),2)
    diff_sum_mean2 += np.power((mean_diff),2)
nse = 1-(diff_sum2/diff_sum_mean2)
print('Nash-Suttcliffe Efficiency', nse)
print('on {} samples'.format(count_samples))
