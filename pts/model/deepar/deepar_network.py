from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution

from pts.core.component import validated
from pts.model import weighted_average
from pts.modules import DistributionOutput, MeanScaler, NOPScaler, FeatureEmbedder


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


class DeepARNetwork(nn.Module):

    @validated()
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        dropout_rate: float,
        cardinality: List[int],
        embedding_dimension: List[int],
        dc_cardinality: List[int],
        dc_embedding_dimension: List[int],
        lags_seq: List[int],
        moving_avg_windows: List[int],
        scaling: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.dc_cardinality = dc_cardinality
        self.dc_embedding_dimension = dc_embedding_dimension
        self.num_cat = len(cardinality)
        self.scaling = scaling
        self.dtype = dtype

        self.lags_seq = lags_seq
        self.moving_avg_windows = moving_avg_windows

        self.distr_output = distr_output
        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[self.cell_type]
        self.rnn = rnn(
            input_size=input_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        '''for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)'''

        self.target_shape = distr_output.event_shape

        self.proj_distr_args = distr_output.get_args_proj(num_cells)

        self.embedder = FeatureEmbedder(
            cardinalities=cardinality, embedding_dims=embedding_dimension
        )
        self.dc_embedder = FeatureEmbedder(
            cardinalities=dc_cardinality, embedding_dims=dc_embedding_dimension
        )

        if scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length : int
            length of sequence in the T (time) dimension (axis = 1).
        indices : List[int]
            list of lag indices to be used.
        subsequences_length : int
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    @staticmethod
    def get_moving_average(
        accumlated_sum: torch.Tensor, 
        moving_avg_windows: List[int],
    ) -> torch.Tensor:
        """
        Returns lagged moving average of a given sequence.
        Parameters
        ----------
        accumlated_sum : Tensor
            the accumulated sum of target sequence.
            Shape: (N, T, C)
        moving_avg_windows: List[int]
            list of window size for averaging
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, T, C, I), where I = len(moving_avg_windows), containing moving_average sequences.
        """
        averaged_seqs = []
        for w in moving_avg_windows:
            moving_avg = torch.zeros_like(accumlated_sum)
            moving_avg[...] = np.nan
            moving_avg[:, w:, ...] = accumlated_sum[:, w:, ...] - accumlated_sum[:, :-w, ...]
            moving_avg /= w
            
            averaged_seqs.append(moving_avg)

        return torch.stack(averaged_seqs, dim=-1)

    def unroll_encoder(
        self,
        feat_static_cat: torch.Tensor,  # (batch_size, num_features)
        feat_static_real: torch.Tensor,  # (batch_size, num_features)
        past_time_feat: torch.Tensor,  # (batch_size, history_length, num_features)
        past_target: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_accumulated_target: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_feat_dynamic_cat: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_feat_dynamic_past: torch.Tensor,  # (batch_size, history_length, *target_shape)
        future_feat_dynamic_past: torch.Tensor,  # (batch_size, history_length, *target_shape)
        future_feat_dynamic_cat: torch.Tensor,  # (batch_size, history_length, *target_shape)
        future_time_feat: Optional[
            torch.Tensor
        ] = None,  # (batch_size, prediction_length, num_features)
        future_target: Optional[
            torch.Tensor
        ] = None,  # (batch_size, prediction_length, *target_shape)
        future_accumulated_target: Optional[torch.Tensor] = None,  # (batch_size, prediction_length, *target_shape)
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, List], torch.Tensor, torch.Tensor]:

        if future_time_feat is None or future_target is None:
            time_feat = past_time_feat[
                :, self.history_length - self.context_length :, ...
            ]
            feat_dynamic_cat = past_feat_dynamic_cat[
                :, self.history_length - self.context_length :, ...
            ]
            feat_dynamic_past = past_feat_dynamic_past
            accumlated_sequence = past_accumulated_target
            sequence = past_target
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (
                    past_time_feat[:, self.history_length - self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1
            )
            feat_dynamic_cat = torch.cat(
                (
                    past_feat_dynamic_cat[:, self.history_length - self.context_length :, ...],
                    future_feat_dynamic_cat,
                ),
                dim=1
            )
            feat_dynamic_past = torch.cat((past_feat_dynamic_past, future_feat_dynamic_past), dim=1)
            accumlated_sequence = torch.cat((past_accumulated_target, future_accumulated_target), dim=1)
            sequence = torch.cat((past_target, future_target), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        #apply lag to feat_dynamic_past for aligning with target-lag
        feat_dynamic_past_lags = self.get_lagged_subsequences(  
            sequence=feat_dynamic_past,
            sequence_length=sequence_length,
            indices=[min(self.lags_seq)],
            subsequences_length=subsequences_length
        ).squeeze(-1)   # (batch_size, subsequences_length, num_features)

        # moving average
        if len(self.moving_avg_windows) == 0:
            merged_sequence = sequence            
        else:
            moving_avg = self.get_moving_average(
                accumlated_sum=accumlated_sequence,
                moving_avg_windows=self.moving_avg_windows
            )
            merged_sequence = torch.cat((sequence.unsqueeze(-1) if len(self.target_shape) == 0 else sequence, moving_avg), dim=-1)

        # apply lags
        lags = self.get_lagged_subsequences(
            sequence=merged_sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length
        )

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags = lags.reshape(
            (-1, subsequences_length, len(self.lags_seq) * (1 + len(self.moving_avg_windows)) * prod(self.target_shape))
        ) # [Note] 모든 lags에 대한 sequence를 생성        

        # embdding dynamic category features
        embedded_dynamic_cat = self.dc_embedder(feat_dynamic_cat)

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)
        _, scale = self.scaler(
            past_target[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        )

        # (batch_size, num_features)
        embedded_cat = self.embedder(feat_static_cat)

        # (batch_size, num_features + prod(target_shape))
        static_feat = torch.cat( (embedded_cat, feat_static_real), dim=1)
        
        if not self.scaling:
            # use the log scale as it can help prediction
            static_feat = torch.cat(
                (
                    static_feat,
                    scale.log() if len(self.target_shape) == 0 else scale.squeeze(1).log(),
                ),
                dim=1,
            )

        # (batch_size, subsequences_length, num_features + 1)
        repeated_static_feat = static_feat.unsqueeze(1).expand(
            -1, subsequences_length, -1
        )

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        input_lags = input_lags / scale.unsqueeze(-1)        

        # (batch_size, sub_seq_len, input_dim)
        inputs = torch.cat((input_lags, time_feat, embedded_dynamic_cat, repeated_static_feat, feat_dynamic_past_lags), dim=-1)

        # unroll encoder
        self.rnn.flatten_parameters()       # resovle warining on multi-gpu training
        outputs, state = self.rnn(inputs)   # [Note] (batch, seq_len, input_size) 순서 batch_first = True로 설정되어 있음

        # outputs: (batch_size, seq_len, num_cells)
        # state: list of (num_layers, batch_size, num_cells) tensors
        # scale: (batch_size, 1, *target_shape)
        # static_feat: (batch_size, num_features + prod(target_shape))
        return outputs, state, scale, static_feat



class RolledDeepARTrainingNetwork(DeepARNetwork):
    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_accumulated_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_feat_dynamic_cat: torch.Tensor,
        past_feat_dynamic_past: torch.Tensor,
        future_feat_dynamic_past: torch.Tensor,
        future_feat_dynamic_cat: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_accumulated_target: torch.Tensor,
        future_observed_values: torch.Tensor
    ) -> torch.Tensor:
        rnn_outputs, state, scale, static_feat = self.unroll_encoder(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_accumulated_target=past_accumulated_target,
            past_observed_values=past_observed_values,
            past_feat_dynamic_cat=past_feat_dynamic_cat,
            past_feat_dynamic_past=past_feat_dynamic_past,
            future_feat_dynamic_past=None,
            future_feat_dynamic_cat=None,
            future_time_feat=None,
            future_target=None,
            future_accumulated_target=None,
        )

        #distr_args = self.proj_distr_args(rnn_outputs)
        #distr = self.distr_output.distribution(distr_args)

        target_scaled = future_target / scale # use normalized target for training

        # blows-up the dimension of each tensor to batch_size * self.num_parallel_samples for increasing parallelism
        repeated_past_target = past_target
        repeated_past_accumulated_target = past_accumulated_target
        repeated_feat_dynamic_past = past_feat_dynamic_past
        embedded_dynamic_cat = self.dc_embedder(future_feat_dynamic_cat)
        repeated_embedded_dynamic_cat = embedded_dynamic_cat
        repeated_time_feat = future_time_feat
        repeated_static_feat = static_feat.unsqueeze(1)
        repeated_scale = scale
        repeated_states = state

        future_samples = []
        losses = []
        self.shifted_lags = [l - 1 for l in self.lags_seq] # [Note] prediction을 위해 1-step forward (0-lag부터 시작
        # for each future time-units we draw new samples for this time-unit and update the state
        for k in range(self.prediction_length):
            feat_dynamic_past_lags = self.get_lagged_subsequences(  
                sequence=repeated_feat_dynamic_past,
                sequence_length=self.history_length + k,
                indices=[min(self.shifted_lags)],
                subsequences_length=1
            ).squeeze(-1)   # (batch_size, 1, num_features)


            if len(self.moving_avg_windows) == 0:
                merged_repeated_past_target = repeated_past_target            
            else:
                # moving average
                repeated_moving_avg = self.get_moving_average(
                    accumlated_sum=repeated_past_accumulated_target,
                    moving_avg_windows=self.moving_avg_windows
                )
                merged_repeated_past_target = torch.cat((repeated_past_target.unsqueeze(-1) if len(self.target_shape) == 0 else repeated_past_target, repeated_moving_avg), dim=-1)

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags = self.get_lagged_subsequences(
                sequence=merged_repeated_past_target,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            # from (batch_size * num_samples, 1, *target_shape, num_lags)
            # to (batch_size * num_samples, 1, prod(target_shape) * num_lags)
            input_lags = lags.reshape(
                (-1, 1, prod(self.target_shape) * len(self.lags_seq)  * (1 + len(self.moving_avg_windows)))
            )

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            input_lags = input_lags / repeated_scale.unsqueeze(-1)

            # (batch_size * num_samples, 1, prod(target_shape) * num_lags + num_time_features + num_static_features)
            decoder_input = torch.cat(
                (input_lags, repeated_time_feat[:, k : k + 1, :], repeated_embedded_dynamic_cat[:, k : k + 1, :], repeated_static_feat, feat_dynamic_past_lags),
                dim=-1,
            )

            # output shape: (batch_size * num_samples, 1, num_cells)
            # state shape: (batch_size * num_samples, num_cells)
            rnn_outputs, repeated_states = self.rnn(decoder_input, repeated_states)

            distr_args = self.proj_distr_args(rnn_outputs)

            # compute likelihood of target given the predicted parameters
            distr = self.distr_output.distribution(distr_args, scale=repeated_scale)

            # (batch_size * num_samples, 1, *target_shape)
            new_samples = distr.sample()

            # (batch_size * num_samples, seq_len, *target_shape)
            repeated_past_target = torch.cat((repeated_past_target, new_samples), dim=1) # [Note] rolling prediction
            future_samples.append(distr.mean)

            # rolling feat_dynamic_past (zero-sale period)            
            future_feat_dynamic_past = repeated_feat_dynamic_past[:,[-1],:] + 1
            future_feat_dynamic_past[new_samples > 0.5] = 0
            repeated_feat_dynamic_past = torch.cat((repeated_feat_dynamic_past, future_feat_dynamic_past), dim=1)   

            # rolling accumulated target
            future_accumulated_target = repeated_past_accumulated_target[:,[-1],...] + new_samples
            repeated_past_accumulated_target = torch.cat((repeated_past_accumulated_target, future_accumulated_target), dim=1)


            # loss
            losses.append(-distr.log_prob(target_scaled[:,k:k+1]))

        loss = torch.cat(losses, dim=1)        
        
        # mask the loss at one time step iff one or more observations is missing in the target dimensions
        # (batch_size, seq_len)
        loss_weights = (
            future_observed_values
            if (len(self.target_shape) == 0)
            else future_observed_values.min(dim=-1, keepdim=False)[0]
        )

        weighted_loss = weighted_average(loss, weights=loss_weights)


        # for mornitoring        
        predicted = torch.cat(future_samples, dim=1)
        true_label = future_target # first true_label (assumed lag-1)
        error = true_label - predicted        

        return weighted_loss, error.detach()

class DeepARPredictionNetwork(DeepARNetwork):
    @validated()
    def __init__(self, num_parallel_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one, at the first time-step
        # of the decoder a lag of one corresponds to the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq] # [Note] prediction을 위해 1-step forward (0-lag부터 시작)

    def sampling_decoder(
        self,
        static_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_accumulated_target: torch.Tensor,
        past_feat_dynamic_past: torch.Tensor,
        time_feat: torch.Tensor,
        dynamic_cat_feat: torch.Tensor,
        scale: torch.Tensor,
        begin_states: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Computes sample paths by unrolling the RNN starting with a initial
        input and state.

        Parameters
        ----------
        static_feat : Tensor
            static features. Shape: (batch_size, num_static_features).
        past_target : Tensor
            target history. Shape: (batch_size, history_length).
        time_feat : Tensor
            time features. Shape: (batch_size, prediction_length, num_time_features).
        scale : Tensor
            tensor containing the scale of each element in the batch. Shape: (batch_size, 1, 1).
        begin_states : List or Tensor
            list of initial states for the LSTM layers or tensor for GRU.
            the shape of each tensor of the list should be (num_layers, batch_size, num_cells)
        Returns
        --------
        Tensor
            A tensor containing sampled paths.
            Shape: (batch_size, num_sample_paths, prediction_length).
        """

        # blows-up the dimension of each tensor to batch_size * self.num_parallel_samples for increasing parallelism
        repeated_past_target = past_target.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_past_accumulated_target = past_accumulated_target.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )        
        repeated_feat_dynamic_past = past_feat_dynamic_past.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        embedded_dynamic_cat = self.dc_embedder(dynamic_cat_feat)
        repeated_embedded_dynamic_cat = embedded_dynamic_cat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_time_feat = time_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_static_feat = static_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        ).unsqueeze(1)
        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        if self.cell_type == "LSTM":
            repeated_states = [
                s.repeat_interleave(repeats=self.num_parallel_samples, dim=1)
                for s in begin_states
            ]
        else:
            repeated_states = begin_states.repeat_interleave(
                repeats=self.num_parallel_samples, dim=1
            )

        future_samples = []

        # for each future time-units we draw new samples for this time-unit and update the state
        for k in range(self.prediction_length):
            feat_dynamic_past_lags = self.get_lagged_subsequences(  
                sequence=repeated_feat_dynamic_past,
                sequence_length=self.history_length + k,
                indices=[min(self.shifted_lags)],
                subsequences_length=1
            ).squeeze(-1)   # (batch_size, 1, num_features)


            if len(self.moving_avg_windows) == 0:
                merged_repeated_past_target = repeated_past_target            
            else:
                # moving average
                repeated_moving_avg = self.get_moving_average(
                    accumlated_sum=repeated_past_accumulated_target,
                    moving_avg_windows=self.moving_avg_windows
                )
                merged_repeated_past_target = torch.cat((repeated_past_target.unsqueeze(-1) if len(self.target_shape) == 0 else repeated_past_target, repeated_moving_avg), dim=-1)

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags = self.get_lagged_subsequences(
                sequence=merged_repeated_past_target,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            # from (batch_size * num_samples, 1, *target_shape, num_lags)
            # to (batch_size * num_samples, 1, prod(target_shape) * num_lags)
            input_lags = lags.reshape(
                (-1, 1, prod(self.target_shape) * len(self.lags_seq)  * (1 + len(self.moving_avg_windows)))
            )

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            input_lags = input_lags / repeated_scale.unsqueeze(-1)

            # (batch_size * num_samples, 1, prod(target_shape) * num_lags + num_time_features + num_static_features)
            decoder_input = torch.cat(
                (input_lags, repeated_time_feat[:, k : k + 1, :], repeated_embedded_dynamic_cat[:, k : k + 1, :], repeated_static_feat, feat_dynamic_past_lags),
                dim=-1,
            )

            # output shape: (batch_size * num_samples, 1, num_cells)
            # state shape: (batch_size * num_samples, num_cells)
            rnn_outputs, repeated_states = self.rnn(decoder_input, repeated_states)

            distr_args = self.proj_distr_args(rnn_outputs)

            # compute likelihood of target given the predicted parameters
            distr = self.distr_output.distribution(distr_args, scale=repeated_scale)

            # (batch_size * num_samples, 1, *target_shape)
            new_samples = distr.sample() # [Note] 샘플링

            # (batch_size * num_samples, seq_len, *target_shape)
            repeated_past_target = torch.cat((repeated_past_target, new_samples), dim=1) # [Note] rolling prediction
            future_samples.append(new_samples)

            # rolling feat_dynamic_past (zero-sale period)            
            future_feat_dynamic_past = repeated_feat_dynamic_past[:,[-1],:] + 1
            future_feat_dynamic_past[new_samples > 0.5] = 0
            repeated_feat_dynamic_past = torch.cat((repeated_feat_dynamic_past, future_feat_dynamic_past), dim=1)   

            # rolling accumulated target
            future_accumulated_target = repeated_past_accumulated_target[:,[-1],...] + new_samples
            repeated_past_accumulated_target = torch.cat((repeated_past_accumulated_target, future_accumulated_target), dim=1)               

        # (batch_size * num_samples, prediction_length, *target_shape)
        samples = torch.cat(future_samples, dim=1)

        # (batch_size, num_samples, prediction_length, *target_shape)
        return samples.reshape(
            (
                (-1, self.num_parallel_samples)
                + (self.prediction_length,)
                + self.target_shape
            )
        )

    # noinspection PyMethodOverriding,PyPep8Naming
    def forward(
        self,
        feat_static_cat: torch.Tensor,  # (batch_size, num_features)
        feat_static_real: torch.Tensor,  # (batch_size, num_features)
        past_time_feat: torch.Tensor,  # (batch_size, history_length, num_features)
        past_target: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_accumulated_target: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_feat_dynamic_cat: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_feat_dynamic_past: torch.Tensor,  # (batch_size, history_length, *target_shape)
        future_feat_dynamic_cat: torch.Tensor,  # (batch_size, history_length, *target_shape)
        future_time_feat: torch.Tensor,  # (batch_size, prediction_length, num_features)
    ) -> torch.Tensor:
        """
        Predicts samples, all tensors should have NTC layout.
        Parameters
        ----------
        feat_static_cat : (batch_size, num_features)
        feat_static_real : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape)
        future_time_feat : (batch_size, prediction_length, num_features)

        Returns
        -------
        Tensor
            Predicted samples
        """

        # unroll the decoder in "prediction mode", i.e. with past data only
        _, state, scale, static_feat = self.unroll_encoder(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_accumulated_target=past_accumulated_target,
            past_observed_values=past_observed_values,
            past_feat_dynamic_cat=past_feat_dynamic_cat,
            past_feat_dynamic_past=past_feat_dynamic_past,
            future_feat_dynamic_past=None,
            future_feat_dynamic_cat=None,
            future_time_feat=None,
            future_target=None,
            future_accumulated_target=None,
        )

        return self.sampling_decoder(
            past_target=past_target,
            past_accumulated_target=past_accumulated_target,
            past_feat_dynamic_past=past_feat_dynamic_past,
            time_feat=future_time_feat,
            dynamic_cat_feat=future_feat_dynamic_cat,
            static_feat=static_feat,
            scale=scale,
            begin_states=state,
        )
