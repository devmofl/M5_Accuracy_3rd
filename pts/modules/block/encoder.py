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
from typing import List, Tuple

# Third-party imports
import torch
import torch.nn as nn
from torch import Tensor

# First-party imports
from pts.core.component import validated
from pts.modules.block.cnn import CausalConv1D
#from pts.modules.block.mlp import MLP
#from pts.modules.block.rnn import RNN


class Seq2SeqEncoder(nn.Module):
    """
    Abstract class for the encoder. An encoder takes a `target` sequence with
    corresponding covariates and maps it into a static latent and
    a dynamic latent code with the same length as the `target` sequence.
    """

    @validated()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # noinspection PyMethodOverriding
    def forward(
        self,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------

        target
            target time series,
            shape (batch_size, sequence_length)

        static_features
            static features,
            shape (batch_size, num_static_features)

        dynamic_features
            dynamic_features,
            shape (batch_size, sequence_length, num_dynamic_features)


        Returns
        -------
        Tensor
            static code,
            shape (batch_size, num_static_features)

        Tensor
            dynamic code,
            shape (batch_size, sequence_length, num_dynamic_features)
        """
        raise NotImplementedError

    @staticmethod
    def _assemble_inputs(
        target: Tensor, static_features: Tensor, dynamic_features: Tensor
    ) -> Tensor:
        """
        Assemble features from target, static features, and the dynamic
        features.

        Parameters
        ----------

        target
            target time series,
            shape (batch_size, sequence_length)

        static_features
            static features,
            shape (batch_size, num_static_features)

        dynamic_features
            dynamic_features,
            shape (batch_size, sequence_length, num_dynamic_features)

        Returns
        -------
        Tensor
            combined features,
            shape (batch_size, sequence_length,
                   num_static_features + num_dynamic_features + 1)

        """
        target = target.unsqueeze(dim=-1)  # (N, T, 1)

        helper_ones = torch.ones_like(target)  # Ones of (N, T, 1)
        tiled_static_features = torch.einsum('bnm, bmk -> bnk', 
            helper_ones, static_features.unsqueeze(1)
        )  # (N, T, C)

        inputs = torch.cat(
            [target, tiled_static_features, dynamic_features], dim=2
        )  # (N, T, C)
        return inputs

    @property    
    def out_channels(self) -> int:
        """
        the size of output channel
        """
        raise NotImplementedError

class HierarchicalCausalConv1DEncoder(Seq2SeqEncoder):
    """
    Defines a stack of dilated convolutions as the encoder.

    See the following paper for details:
    1. Van Den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner,
    N., Senior, A.W. and Kavukcuoglu, K., 2016, September. WaveNet: A generative model for raw audio. In SSW (p. 125).

    Parameters
    ----------
    dilation_seq
        dilation for each convolution in the stack.

    kernel_size_seq
        kernel size for each convolution in the stack.

    channels_seq
        number of channels for each convolution in the stack.

    use_residual
        flag to toggle using residual connections.

    use_covariates
        flag to toggle whether to use coveriates as input to the encoder
    """

    @validated()
    def __init__(
        self,
        input_size: int,
        dilation_seq: List[int],
        kernel_size_seq: List[int],        
        channels_seq: List[int],
        use_residual: bool = False,
        use_covariates: bool = False,
        **kwargs,
    ) -> None:
        assert all(
            [x > 0 for x in dilation_seq]
        ), "`dilation_seq` values must be greater than zero"
        assert all(
            [x > 0 for x in kernel_size_seq]
        ), "`kernel_size_seq` values must be greater than zero"
        assert all(
            [x > 0 for x in channels_seq]
        ), "`channel_dim_seq` values must be greater than zero"

        super().__init__(**kwargs)

        self.use_residual = use_residual
        self.use_covariates = use_covariates
        self.CNNs = nn.Sequential()
        self.last_out_channel = channels_seq[-1]

        it = zip(channels_seq, kernel_size_seq, dilation_seq)
        in_channels = input_size
        for layer_no, (out_channels, kernel_size, dilation) in enumerate(it):
            convolution = CausalConv1D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                activation="relu",
            )
            self.CNNs.add_module('conv_%02d' % (layer_no), convolution)
            in_channels = out_channels

    def forward(
        self,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------

        target
            target time series,
            shape (batch_size, sequence_length)

        static_features
            static features,
            shape (batch_size, num_static_features)

        dynamic_features
            dynamic_features,
            shape (batch_size, sequence_length, num_dynamic_features)

        Returns
        -------
        Tensor
            static code,
            shape (batch_size, num_static_features)

        Tensor
            dynamic code,
            shape (batch_size, sequence_length, num_dynamic_features)
        """

        if self.use_covariates:
            inputs = Seq2SeqEncoder._assemble_inputs(
                target=target,
                static_features=static_features,
                dynamic_features=dynamic_features,
            )
        else:
            inputs = target

        # NTC -> NCT (or NCW)
        ct = inputs.transpose(1, 2)
        ct = self.CNNs(ct)

        # now we are back in NTC
        ct = ct.transpose(1, 2)
        
        if self.use_residual:
            ct = torch.cat([ct, target.unsqueeze(-1)], dim=2)

        # return the last state as the static code
        static_code = ct[:, -1:, ...]
        static_code = torch.squeeze(static_code, dim=1)
        return static_code, ct

    @property    
    def out_channels(self) -> int:
        """
        the size of output channel
        """
        return self.last_out_channel + 1

'''
class RNNEncoder(Seq2SeqEncoder):
    """
    Defines an RNN as the encoder.

    Parameters
    ----------
    mode
        type of the RNN. Can be either: rnn_relu (RNN with relu activation),
        rnn_tanh, (RNN with tanh activation), lstm or gru.

    hidden_size
        number of units per hidden layer.

    num_layers
        number of hidden layers.

    bidirectional
        toggle use of bi-directional RNN as encoder.
    """

    @validated()
    def __init__(
        self,
        mode: str,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        **kwargs,
    ) -> None:
        assert num_layers > 0, "`num_layers` value must be greater than zero"
        assert hidden_size > 0, "`hidden_size` value must be greater than zero"

        super().__init__(**kwargs)

        self.rnn = RNN(mode, hidden_size, num_layers, bidirectional)

    def forward(
        self,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------

        target
            target time series,
            shape (batch_size, sequence_length)

        static_features
            static features,
            shape (batch_size, num_static_features)

        dynamic_features
            dynamic_features,
            shape (batch_size, sequence_length, num_dynamic_features)

        Returns
        -------
        Tensor
            static code,
            shape (batch_size, num_static_features)

        Tensor
            dynamic code,
            shape (batch_size, sequence_length, num_dynamic_features)
        """
        dynamic_code = self.rnn(target)
        static_code = dynamic_code[:, -1:, ...]
        return static_code, dynamic_code


class MLPEncoder(Seq2SeqEncoder):
    """
    Defines a multilayer perceptron used as an encoder.

    Parameters
    ----------
    layer_sizes
        number of hidden units per layer.
    kwargs
    """

    @validated()
    def __init__(self, layer_sizes: List[int], **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = MLP(layer_sizes, flatten=True)

    def forward(
        self,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------

        target
            target time series,
            shape (batch_size, sequence_length)

        static_features
            static features,
            shape (batch_size, num_static_features)

        dynamic_features
            dynamic_features,
            shape (batch_size, sequence_length, num_dynamic_features)

        Returns
        -------
        Tensor
            static code,
            shape (batch_size, num_static_features)

        Tensor
            dynamic code,
            shape (batch_size, sequence_length, num_dynamic_features)
        """

        inputs = Seq2SeqEncoder._assemble_inputs(
            target, static_features, dynamic_features
        )
        static_code = self.model(inputs)
        dynamic_code = torch.zeros_like(target).unsqueeze(2)
        return static_code, dynamic_code


class RNNCovariateEncoder(Seq2SeqEncoder):
    """
    Defines RNN encoder that uses covariates and target as input to the RNN.

    Parameters
    ----------
    mode
        type of the RNN. Can be either: rnn_relu (RNN with relu activation),
        rnn_tanh, (RNN with tanh activation), lstm or gru.

    hidden_size
        number of units per hidden layer.

    num_layers
        number of hidden layers.

    bidirectional
        toggle use of bi-directional RNN as encoder.
    """

    @validated()
    def __init__(
        self,
        mode: str,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        **kwargs,
    ) -> None:

        assert num_layers > 0, "`num_layers` value must be greater than zero"
        assert hidden_size > 0, "`hidden_size` value must be greater than zero"

        super().__init__(**kwargs)

        self.rnn = RNN(mode, hidden_size, num_layers, bidirectional)

    def forward(
        self,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------

        target
            target time series,
            shape (batch_size, sequence_length)

        static_features
            static features,
            shape (batch_size, num_static_features)

        dynamic_features
            dynamic_features,
            shape (batch_size, sequence_length, num_dynamic_features)

        Returns
        -------
        Tensor
            static code,
            shape (batch_size, num_static_features)

        Tensor
            dynamic code,
            shape (batch_size, sequence_length, num_dynamic_features)
        """
        inputs = Seq2SeqEncoder._assemble_inputs(
            target, static_features, dynamic_features
        )
        dynamic_code = self.rnn(inputs)

        # using the last state as the static code,
        # but not working as well as the concat of all the previous states
        static_code = torch.squeeze(dynamic_code[:, -1:, ...], dim=1)

        return static_code, dynamic_code
'''