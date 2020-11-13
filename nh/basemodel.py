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

class BaseModel(nn.Module):
    """Abstract base model class, don't use this class for model training.
    Use subclasses of this class for training/evaluating different models, e.g. use `CudaLSTM` for training a standard
    LSTM model or `EA-LSTM` for training an Entity-Aware-LSTM. Refer to
    `Documentation/Modelzoo <https://neuralhydrology.readthedocs.io/en/latest/usage/models.html>`_ for a full list of
    available models and how to integrate a new model. 
    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, xconfigx):
        super(BaseModel, self).__init__()
        self.cfg = xconfigx
        print(xconfigx)
        self.output_size = len(xconfigx['target_variables'])


    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass.
        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.
        Returns
        -------
        Dict[str, torch.Tensor]
            Model output and potentially any intermediate states and activations as a dictionary.
        """
        raise NotImplementedError

    def sample(self, data: Dict[str, torch.Tensor], n_samples: int) -> torch.Tensor:
        """Sample model predictions, e.g., for MC-Dropout.
        
        This function does `n_samples` forward passes for each sample in the batch. Only useful for models with dropout,
        to perform MC-Dropout sampling. Make sure to set the model to train mode before calling this function 
        (`model.train()`), otherwise dropout won't be active.
        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.
        n_samples : int
            Number of samples to generate for each input sample.
        Returns
        -------
        torch.Tensor
            Sampled model outputs for the `predict_last_n` (config argument) time steps of each sequence. The shape of 
            the output is ``[batch size, predict_last_n, n_samples]``.
        """
        predict_last_n = xconfigx['predict_last_n']
        samples = torch.zeros(data['x_d'].shape[0], predict_last_n, n_samples)
        for i in range(n_samples):
            prediction = self.forward(data)
            samples[:, -predict_last_n:, i] = prediction['y_hat'][:, -predict_last_n:, 0]

        return samples