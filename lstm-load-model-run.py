import numpy as np
import pandas as pd
import torch
from torch import nn
import pickle
import matplotlib as plt

data_dir = './data/'
torch.manual_seed(0)

model = torch.load(data_dir+'sugar_creek_IL_trained_model.pt')

batch_size=1
hidden_layer_size=64
seq_length = 336
istart=71593
iend=72338

with open(data_dir+'sugar_creek_input_all2.csv','r') as f:
    df = pd.read_csv(f)
#df = df.drop(['date','obs'], axis=1)

with open(data_dir+'sugar_creek_scaler.p', 'rb') as fb:
    scalers = pickle.load(fb)
with open(data_dir+'sugar_creek_configs.pickle', 'rb') as fb:
    cfg = pickle.load(fb)
with  open(data_dir+'sugar_creek_basin_data.csv', 'r') as f:
    sug_crek = pd.read_csv(f)
obs = list(sug_crek.iloc[istart:iend,-1])

obs_std = np.array(scalers['xarray_stds'].to_array())[-1]
obs_mean = np.array(scalers['xarray_means'].to_array())[-1]
att_means = np.append( np.array(scalers['attribute_means'].values)[0],np.array(scalers['attribute_means'].values)[1:3])
att_stds = np.append(np.array(scalers['attribute_stds'].values)[0], np.array(scalers['attribute_stds'].values)[1:3])
scaler_mean = np.array(scalers['xarray_means'].to_array())[:-1]
scaler_std = np.array(scalers['xarray_stds'].to_array())[:-1]

with torch.no_grad():
    x_d = ((torch.tensor(df.loc[istart-seq_length:iend,['LWDOWN', 'PSFC', 'Q2D', 'RAINRATE', 'SWDOWN', 'T2D', 'U2D', 'V2D']].values)-scaler_mean)/scaler_std).unsqueeze(0).float()
    x_s = ((torch.tensor(df.loc[istart-seq_length:iend,['area_sqkm', 'lat', 'lon']].values)-att_means)/att_stds).unsqueeze(0)[:,0,:].float()
    input_layer = {'x_d':x_d, 'x_s':x_s}
    output = model(input_layer)

output = (output['y_hat'].numpy().flatten() * obs_std + obs_mean).tolist()[seq_length+1:]

print('output stats')
print('mean', np.mean(output))
print('min', np.min(output))
print('max', np.max(output))
print('observation stats')
print('mean', np.nanmean(obs))
print('min', np.nanmin(obs))
print('max', np.nanmax(obs))

diff_sum2 = 0
diff_sum_mean2 = 0
obs_mean = np.nanmean(np.array(obs))
count_samples = 0
for j, k in zip(output, obs):
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
