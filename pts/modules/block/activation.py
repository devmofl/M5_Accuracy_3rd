from typing import Optional, Union, List, Tuple

# Third-party imports
import torch.nn as nn
from torch import Tensor

class Activation(nn.Module):
    """
    Activation fuction

    Parameters
    ----------

    activation
        Activation function to use.

    """

    def __init__(
        self,
        activation: Optional[str] = "identity",
    ):
        super(Activation, self).__init__()

        activations = {
            'identity': nn.Identity(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        self.activation = activations[activation]
    
    def forward(self, data: Tensor) -> Tensor:
        """
        applying activation function.


        Parameters
        ----------
        data
            Shape : any shape of tensor

        Returns
        -------
        Tensor
            activation(x). Shape is the same with input
        """        
        return self.activation(data)