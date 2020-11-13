import logging
from typing import Dict

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