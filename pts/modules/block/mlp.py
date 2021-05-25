# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
from typing import List

# Third-party imports
import torch.nn as nn
from torch import Tensor

# First-party imports
from pts.core.component import validated
from pts.modules.block.activation import Activation

class MLP(nn.Module):
    """
    Defines an MLP block.

    Parameters
    ----------
    layer_sizes
        number of hidden units per layer.

    flatten
        toggle whether to flatten the output tensor.

    activation
        activation function of the MLP, default is relu.
    """

    @validated()
    def __init__(
        self, input_size, layer_sizes: List[int], activation="relu"
    ) -> None:
        super().__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.Sequential()

        in_channel = input_size
        for layer_no, layer_dim in enumerate(layer_sizes):
            self.layers.add_module(
                'linear_%02d' % (layer_no),
                nn.Linear(in_channel, layer_dim)
            )
            self.layers.add_module('%s_%02d' % (activation, layer_no), Activation(activation))
                
            in_channel = layer_dim

    # noinspection PyMethodOverriding
    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
        Tensor
            Output of the MLP given the input tensor.
        """
        return self.layers(x)
