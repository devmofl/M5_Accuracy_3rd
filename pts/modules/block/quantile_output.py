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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Third-party imports


# First-party imports
from pts.core.component import validated



class QuantileLoss(nn.Module):
    @validated()
    def __init__(
        self,
        quantiles: List[float],
        quantile_weights: List[float] = None,
        **kwargs,
    ) -> None:
        """
        Represents the quantile loss used to fit decoders that learn quantiles.

        Parameters
        ----------
        quantiles
            list of quantiles to compute loss over.

        quantile_weights
            weights of the quantiles.
        """
        super().__init__(**kwargs)

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.quantile_weights = (
            torch.ones(self.num_quantiles) / self.num_quantiles
            if not quantile_weights
            else quantile_weights
        )

    # noinspection PyMethodOverriding
    def forward(
        self, y_true: Tensor, y_pred: Tensor, sample_weight=None
    ):
        """
        Compute the weighted sum of quantile losses.

        Parameters
        ----------
        y_true
            true target, shape (N1 x N2 x ... x Nk x dimension of time series
            (normally 1))
        y_pred
            predicted target, shape (N1 x N2 x ... x Nk x num_quantiles)
        sample_weight
            sample weights

        Returns
        -------
        Tensor
            weighted sum of the quantile losses, shape N1 x N1 x ... Nk
        """
        y_pred_all = y_pred.chunk(self.num_quantiles, dim=-1)

        qt_loss = []
        for i, y_pred_q in enumerate(y_pred_all):
            q = self.quantiles[i]
            weighted_qt = (
                self.compute_quantile_loss(y_true, y_pred_q.squeeze(-1), q)
                * self.quantile_weights[i].detach()
            )
            qt_loss.append(weighted_qt)
        stacked_qt_losses = torch.stack(qt_loss, axis=-1)
        sum_qt_loss = torch.mean(
            stacked_qt_losses, axis=-1
        )  # avg across quantiles
        if sample_weight is not None:
            return sample_weight * sum_qt_loss
        else:
            return sum_qt_loss

    @staticmethod
    def compute_quantile_loss(
        y_true: Tensor, y_pred_p: Tensor, p: float
    ) -> Tensor:
        """
        Compute the quantile loss of the given quantile

        Parameters
        ----------

        y_true
            true target, shape (N1 x N2 x ... x Nk x dimension of time series
            (normally 1)).

        y_pred_p
            predicted target quantile, shape (N1 x N2 x ... x Nk x 1).

        p
            quantile error to compute the loss.

        Returns
        -------
        Tensor
            quantile loss, shape: (N1 x N2 x ... x Nk x 1)
        """

        under_bias = p * F.relu(y_true - y_pred_p)
        over_bias = (1 - p) * F.relu(y_pred_p - y_true)

        qt_loss = 2 * (under_bias + over_bias)

        return qt_loss


class ProjectParams(nn.Module):
    """
    Defines a dense layer to compute the projection weights into the quantile
    space.

    Parameters
    ----------
    num_quantiles
        number of quantiles to compute the projection.
    """

    @validated()
    def __init__(self, input_size, num_quantiles, **kwargs):
        super().__init__(**kwargs)

        self.projection = nn.Linear(input_size, num_quantiles)

    # noinspection PyMethodOverriding,PyPep8Naming
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
        Tensor
            output of the projection layer
        """
        return self.projection(x)


class QuantileOutput:
    """
    Output layer using a quantile loss and projection layer to connect the
    quantile output to the network.

    Parameters
    ----------
        quantiles
            list of quantiles to compute loss over.

        quantile_weights
            weights of the quantiles.
    """

    @validated()
    def __init__(
        self,
        input_size: int,
        quantiles: List[float],
        quantile_weights: Optional[List[float]] = None,
    ) -> None:
        self.input_size = input_size
        self.quantiles = quantiles
        self.quantile_weights = quantile_weights        

    def get_loss(self) -> Tensor:
        """
        Returns
        -------
        nn.Module
            constructs quantile loss object.
        """
        return QuantileLoss(
            quantiles=self.quantiles, quantile_weights=self.quantile_weights
        )

    def get_quantile_proj(self, **kwargs) -> nn.Module:
        """
        Returns
        -------
        nn.Module
            constructs projection parameter object.

        """
        return ProjectParams(self.input_size, len(self.quantiles), **kwargs)
