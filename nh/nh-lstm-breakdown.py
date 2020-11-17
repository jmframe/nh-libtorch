import logging
from typing import Dict
import pickle

import torch
import torch.nn as nn

#from config import Config
xconfigx = {
'experiment_name': 'nwmv3_test_run',
'initial_forget_bias':3,
'hidden_size': 64,
'dynamic_inputs': ['RAINRATE','Q2D','T2D','LWDOWN','SWDOWN','PSFC','U2D','V2D'],
'embedding_hiddens': [30,20,64],
'camels_attributes': ['lat','lon','area_sqkm'],
'static_inputs': None,
'hydroatlas_attributes': None,
'number_of_basins': 8,
'use_basin_id_encoding': False,
'predict_last_n':1,
'head':'regression',
'target_variables': 'obs',
'embedding_dropout': 0.0,
'output_dropout': 0.4,
'embedding_activation': 'tanh',
'output_activation': 'linear'
}

with open('sugar_creek_data.p','rb') as fp:
    ds = pickle.load(fp)

# At some point I have to load in the trained_lstm...
#this_nn = nn.Module.load_state_dict(torch.load('trained_lstm.pt', map_location=torch.device(device)))

LOGGER = logging.getLogger(__name__)


class nhLSTM(nn.Module):
    """LSTM model class, which relies on PyTorch's CUDA LSTM class.

    This class implements the standard LSTM combined with a model head, as specified in the config. All features 
    (time series and static) are concatenated and passed to the LSTM directly. If you want to embedd the static features
    prior to the concatenation, use the `EmbCudaLSTM` class.
    To control the initial forget gate bias, use the config argument `initial_forget_bias`. Often it is useful to set 
    this value to a positive value at the start of the model training, to keep the forget gate closed and to facilitate
    the gradient flow. 
    The `CudaLSTM` class does only support single timescale predictions. Use `MTSLSTM` to train a model and get 
    predictions on multiple temporal resolutions at the same time.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, xconfigx):
        super(nhLSTM, self).__init__()

        input_size = len(xconfigx['dynamic_inputs'] + xconfigx['camels_attributes'])
        if xconfigx['use_basin_id_encoding']:
            input_size += xconfigx['number_of_basin']

        if xconfigx['head'].lower() == "umal":
            input_size += 1

        nn.Module.load_state_dict(torch.load('trained_lstm.pt', map_location=self.device(device)))

        self.output_size = len(xconfigx['target_variables'])

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=xconfigx['hidden_size'])

        self.dropout = nn.Dropout(p=xconfigx['output_dropout'])

        #self.head = get_head(xconfigx,n_in=xconfigx['hidden_size'], n_out=self.output_size)
        layers = [nn.Linear(xconfigx['hidden_size'], self.output_size)]
        dotnet = nn.Sequential(*layers)
        self.head = {'y_hat': dotnet(x)}

        if xconfigx['initial_forget_bias'] is not None:
            self.lstm.bias_hh_l0.data[xconfigx['hidden_size']:2 * xconfigx['hidden_size']] = \
                xconfigx['initial_forget_bias']

    def forward(self, ds) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the CudaLSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary. 
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [1, batch size, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [1, batch size, hidden size].
        """
        # transpose to [seq_length, batch_size, n_features]
        x_d = data['x_d'].transpose(0, 1)

        # concat all inputs
        x_s = data['x_s'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
        x_d = torch.cat([x_d, x_s], dim=-1)

        lstm_output, (h_n, c_n) = self.lstm(input=x_d)

        # reshape to [1 , batch_size, n_hiddens]
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        pred = {'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(lstm_output.transpose(0, 1))))

        return pred

#############################################
#####    CONVERT TO LIBTORCH WITH JIT   #####
#############################################
model = nhLSTM(xconfigx)

with torch.no_grad():
    # Generate a bunch of fake dta to feed the model when we 'torch.jit.script' it
    # since it is needed by the JIT (not sure why)
    fake_input = torch.zeros((10,100))

    # Trace the model using 'torch.jit.script'
    traced = torch.jit.script(model, fake_input)

    # Print the Torch Script code
    print(traced.code)

    # We can also store the model like usual:
    traced.save('nhlstm.ptc')
