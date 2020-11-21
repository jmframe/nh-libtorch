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
'target_variables': ['obs'],
'embedding_dropout': 0.0,
'output_dropout': 0.4,
'embedding_activation': 'tanh',
'output_activation': 'linear'
}

with open('sugar_creek_data.p','rb') as fp:
    ds = pickle.load(fp)
with open('sugar_creek_scaler.p','rb') as fp:
    scalar = pickle.load(fp)
with open('sugar_creek_trained.pt','r') as f:
    trained_model = f 

LOGGER = logging.getLogger(__name__)

def get_head(xconfigx, n_in: int, n_out: int) -> nn.Module:
    """Get specific head module, depending on the run configuration.
    Parameters
    ----------
    cfg : Config
        The run configuration.
    n_in : int
        Number of input features.
    n_out : int
        Number of output features.
    Returns
    -------
    nn.Module
        The model head, as specified in the run configuration.
    """
    if xconfigx['head'].lower() == "regression":
        head = Regression(n_in=n_in, n_out=n_out, activation=xconfigx['output_activation'])
    else:
        raise NotImplementedError(f"{xconfigx['head']} not implemented or not linked in `get_head()`")

    return head


class Regression(nn.Module):
    """Single-layer regression head with different output activations.
    
    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons.
    activation: str, optional
        Output activation function. Can be specified in the config using the `output_activation` argument. Supported
        are {'linear', 'relu', 'softplus'}. If not specified (or an unsupported activation function is specified), will
        default to 'linear' activation.
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "linear"):
        super(Regression, self).__init__()

        # TODO: Add multi-layer support
        layers = [nn.Linear(n_in, n_out)]
        if activation != "linear":
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "softplus":
                layers.append(nn.Softplus())
            else:
                LOGGER.warning(f"## WARNING: Ignored output activation {activation} and used 'linear' instead.")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the Regression head.
        
        Parameters
        ----------
        x : torch.Tensor
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary, containing the model predictions in the 'y_hat' key.
        """
        return {'y_hat': self.net(x)}

class nhLSTM(nn.Module):

    def __init__(self, xconfigx):
        super(nhLSTM, self).__init__()

        input_size = len(xconfigx['dynamic_inputs'] + xconfigx['camels_attributes'])
        if xconfigx['use_basin_id_encoding']:
            input_size += xconfigx['number_of_basin']

        if xconfigx['head'].lower() == "umal":
            input_size += 1

        self.output_size = len(xconfigx['target_variables'])

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=xconfigx['hidden_size'])

        self.dropout = nn.Dropout(p=xconfigx['output_dropout'])
        
        # replacing the get_head() function
        layers = [nn.Linear(xconfigx['hidden_size'], self.output_size)]
        dotnet = nn.Sequential(*layers)
        self.head = get_head(xconfigx,n_in=xconfigx['hidden_size'], n_out=self.output_size)

        if xconfigx['initial_forget_bias'] is not None:
            self.lstm.bias_hh_l0.data[xconfigx['hidden_size']:2 * xconfigx['hidden_size']] = \
                xconfigx['initial_forget_bias']

    def forward(self, ds) -> Dict[str, torch.Tensor]:
        # transpose to [seq_length, batch_size, n_features]
        x_d = ds['x_d']
        x_d = x_d.unsqueeze(1)

        # concat all inputs
        x_s = ds['x_s'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
        x_d = torch.cat([x_d, x_s], dim=-1)

        lstm_output, (h_n, c_n) = self.lstm(input=x_d)

        # reshape to [1 , batch_size, n_hiddens]
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        pred = {'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(lstm_output.transpose(0, 1))))

        return pred


# Set the model
model = nhLSTM(xconfigx)
#for i in model.state_dict():
#    print(i)
#for i in torch.load('sugar_creek_trained.pt'):
#    print(i)
    # Load weights
#    model.load_state_dict(torch.load('sugar_creek_trained.pt'))
#    

pretrained_dict = torch.load('sugar_creek_trained.pt')
model_dict = model.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# # 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# # 3. load the new state dict
model.load_state_dict(pretrained_dict)

for data in ds:
    y_hat = model(data)
    print(y_hat['y_hat'][0][0])

#y_hat = model(ds)
#    
#    # rescale predictions
#    y_hat = y_hat * scaler["xarray_stds"][xconfigx['target_variables']].to_array().values
#    y_hat_freq = y_hat + self.scaler["xarray_means"][xconficx['target_variables']].to_array().values


#############################################
#####    CONVERT TO LIBTORCH WITH JIT   #####
#############################################
#   
#   with torch.no_grad():
#       # Generate a bunch of fake dta to feed the model when we 'torch.jit.script' it
#       # since it is needed by the JIT (not sure why)
#       fake_input = torch.zeros((10,100))
#   
#       # Trace the model using 'torch.jit.script'
#       traced = torch.jit.script(model, fake_input)
#   
#       # Print the Torch Script code
#       print(traced.code)
#   
#       # We can also store the model like usual:
#       traced.save('nhlstm.ptc')
