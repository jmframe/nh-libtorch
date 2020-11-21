import torch
from torch import nn

torch.manual_seed(0)

"""
clas Sequence;

Model capable of doing sine wave prediction. We don't care about the model itself for now,
just assume it works.

Source:
  https://github.com/pytorch/examples/blob/master/time_sequence_prediction/train.py 
"""
class Sequence(nn.Module):

    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(11,51)
        self.lstm2 = nn.LSTMCell(51,51)
        self.linear = nn.Linear(51,11)

        self.init_func = 'zeros'

    def forward(self, input, future):
        outputs = []

        # Changed 'torch.double' to 'torch.float' because the LSTM layer will complain
        if self.init_func == 'ones':
            h_t = torch.ones(input.size(0),51, dtype=torch.float)
            c_t = torch.ones(input.size(0),51, dtype=torch.float)
            h_t2 = torch.ones(input.size(0),51, dtype=torch.float)
            c_t2 = torch.ones(input.size(0),51, dtype=torch.float)
            output = torch.ones((1,1))
        else:
            h_t = torch.zeros(input.size(0),51, dtype=torch.float)
            c_t = torch.zeros(input.size(0),51, dtype=torch.float)
            h_t2 = torch.zeros(input.size(0),51, dtype=torch.float)
            c_t2 = torch.zeros(input.size(0),51, dtype=torch.float)
            output = torch.zeros((1,1))

        for i, input_t in enumerate(input.chunk(input.size(1), dim=11)):
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

model = Sequence()

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
