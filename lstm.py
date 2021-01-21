import torch
from torch import nn
import pickle

torch.manual_seed(0)

class LSTM(nn.Module):
    def __init__(self, input_size=11, hidden_layer_size=64, output_size=1, batch_size=1, seq_length=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.seq_length=seq_length
        self.batch_size = batch_size # We shouldn't neeed to do a higher batch size.
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.head = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_layer, h_t, c_t):
        h_t = h_t.float()
        c_t = c_t.float()
        input_layer = input_layer.float()
        input_view = input_layer.view(self.seq_length, self.batch_size, self.input_size)
        output, (h_t, c_t) = self.lstm(input_view, (h_t,c_t))
        prediction = self.head(output)
        return prediction, h_t, c_t

model = LSTM()

data_dir = './data/'
pretrained_dict = torch.load(data_dir+'nwmv3_normalarea_trained.pt', map_location=torch.device('cpu'))
pretrained_dict['head.weight'] = pretrained_dict.pop('head.net.0.weight')
pretrained_dict['head.bias'] = pretrained_dict.pop('head.net.0.bias')
model.load_state_dict(pretrained_dict)

with torch.no_grad():
    # Generate a bunch of fake dta to feed the model when we 'torch.jit.script' it
    # since it is needed by the JIT (not sure why)
    inp = torch.zeros((1,11))
    h_t = torch.zeros((1,1,64))
    c_t = torch.zeros((1,1,64))

    # Trace the model using 'torch.jit.script'
    traced = torch.jit.script(model, (inp, h_t, c_t))

    # Print the Torch Script code
    print(traced.code)

    # We can also store the model like usual:
    traced.save('lstm.ptc')
