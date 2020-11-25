import numpy as np
import pandas as pd
import torch
from torch import nn
import pickle

torch.manual_seed(0)


"""
clas Sequence;

"""
class Sequence(nn.Module):

    def __init__(self):
        super(Sequence, self).__init__()
        self.n = 51
        n = self.n
        self.lstm1 = nn.LSTMCell(11,n)
        self.lstm2 = nn.LSTMCell(n,n)
        self.linear = nn.Linear(n,1)

    def forward(self, input, future):
        outputs = []
        m = input.size(1)
        n = self.n
        h_t = torch.zeros(m,n, dtype=torch.float)
        c_t = torch.zeros(m,n, dtype=torch.float)
        h_t2 = torch.zeros(m,n, dtype=torch.float)
        c_t2 = torch.zeros(m,n, dtype=torch.float)
        output = torch.zeros((1,1))
        print('starting loop')
        for i, input_t in enumerate(input):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for i in range(int(future)):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs

# Now try running the model in Python, because if we can't run in python, we have no chance in C++
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
#input_tensor = input_tensor.unsqueeze(0)
input_dict = {'input':input_tensor[0:300,:],'future':input_tensor[301,:]}


class LSTM(nn.Module):
    def __init__(self, input_size=11, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = 1 # In this application we do one timestep only
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_layer, init_states=None):
        if init_states is None:
            h_t, c_t = (torch.zeros(self.batch_size, self.hidden_layer_size),
                        torch.zeros(self.batch_size, self.hidden_layer_size))
        else:
            h_t, c_t = init_states
        h_t = h_t.float()
        c_t = c_t.float()
        input_layer = input_layer.float()
        input_view = input_layer.view(1,1,-1)
        output, (c_t, h_t) = self.lstm(input_view, (h_t,c_t))
        prediction = self.linear(h_t)
        return prediction, (h_t, c_t)

model = LSTM()

print(model)

hidden_layer_size = 100
states = (torch.zeros(1,1, hidden_layer_size).float(),
          torch.zeros(1,1, hidden_layer_size).float())
for t in range(300):
    output, states = model(input_dict['input'][t,:], states)       
    print(output)






