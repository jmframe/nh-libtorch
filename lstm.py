import torch
from torch import nn

torch.manual_seed(0)

class LSTM(nn.Module):
    def __init__(self, input_size=11, hidden_layer_size=100, output_size=1):
        super().__init__()
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

with torch.no_grad():
    # Generate a bunch of fake dta to feed the model when we 'torch.jit.script' it
    # since it is needed by the JIT (not sure why)
    fake_input = torch.zeros((11,100))

    # Trace the model using 'torch.jit.script'
    traced = torch.jit.script(model, fake_input)

    # Print the Torch Script code
    print(traced.code)

    # We can also store the model like usual:
    traced.save('lstm.ptc')
