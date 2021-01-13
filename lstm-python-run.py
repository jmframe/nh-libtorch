import numpy as np
import pandas as pd
import torch
from torch import nn
import pickle
import matplotlib as plt
import nwmv3_get_data

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

istart=337    # 71593 <- this number was for the NWMV3 data
iend=1057
warmup = np.maximum(seq_length, 336)
do_warmup = True

data_dir = './data/'

if False:
    b = '03500240' #'03471500'
    df = nwmv3_get_data.dynamic_data(b)
    att = nwmv3_get_data.static_data()
    att = att.loc[int(b), ['area_sqkm','lat','lon']]
    df['area_sqkm'] = att['area_sqkm']
    df['lat'] = att['lat']
    df['lon'] = att['lon']
    df = df.loc['2015-11-17':'2015-12-30']
    obs = list(df.loc['2015-12-01':'2015-12-30' , 'obs'])
    df = df.loc[:,['RAINRATE', 'Q2D', 'T2D', 'LWDOWN',  'SWDOWN',  'PSFC',  'U2D', 'V2D', 'area_sqkm', 'lat', 'lon']]
    output_factor = df['area_sqkm'][0]
else:
    with open(data_dir+'cat-87-forcing.csv','r') as f:
        df = pd.read_csv(f)
    df = df.drop(['date'], axis=1) #cat-87.csv has no observation data
    df = df.loc[:,['RAINRATE', 'Q2D', 'T2D', 'LWDOWN',  'SWDOWN',  'PSFC',  'U2D', 'V2D', 'area_sqkm', 'lat', 'lon']]
    df['area_sqkm'] = 14.8
    output_factor = df['area_sqkm'][0] * 35.315
    # The precipitation rate units for the training set were obviously different than these forcings. Guessing it is a m -> mm conversion.
    df['RAINRATE'] = df['RAINRATE']*1000
    with open(data_dir+'obs_q_02146562.csv', 'r') as f:
        obs = pd.read_csv(f)
    obs = pd.Series(data=list(obs['90100_00060']), index=pd.to_datetime(obs.datetime)).resample('60T').mean()
    obs = obs.loc['2015-12-01':'2015-12-30']

input_tensor = torch.tensor(df.values)

#with open(data_dir+'nwmv3_scaler.p', 'rb') as fb:
with open(data_dir+'nwmv3_normalarea_scaler.p', 'rb') as fb:
    scalers = pickle.load(fb)

#p_dict = torch.load(data_dir+'nwmv3_trained.pt', map_location=torch.device('cpu'))
p_dict = torch.load(data_dir+'nwmv3_normalarea_trained.pt', map_location=torch.device('cpu'))
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
xm2 = np.array([xm[x] for x in list([3,2,5,0,4,1,6,7])]) # scalar means ordered to match forcing
xs2 = np.array([xs[x] for x in list([3,2,5,0,4,1,6,7])]) # scalar stds ordered to match forcing
att_means = np.append( np.array(scalers['attribute_means'].values)[0],np.array(scalers['attribute_means'].values)[1:3])
att_stds = np.append(np.array(scalers['attribute_stds'].values)[0], np.array(scalers['attribute_stds'].values)[1:3])
scaler_mean = np.append(xm2, att_means)
scaler_std = np.append(xs2, att_stds)
input_tensor = (input_tensor-scaler_mean)/scaler_std

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
        output = (output[0,0,0].numpy().tolist() * obs_std + obs_mean) * output_factor
        output_list.append(output)
#        print(input_layer)
        print(output)

print('output stats')
print('mean', np.mean(output_list))
print('min', np.min(output_list))
print('max', np.max(output_list))
print('observation stats')
print('mean', np.nanmean(obs))
print('min', np.nanmin(obs))
print('max', np.nanmax(obs))

print('length obs', len(obs))
print('length output', len(output_list))

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

