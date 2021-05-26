from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from pts.core.component import validated
from pts import Trainer
from pts.dataset import FieldName
from pts.feature import (
    TimeFeature,
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from pts.model import PTSEstimator, Predictor, PTSPredictor, copy_parameters
from pts.modules import DistributionOutput, StudentTOutput
from pts.transform import (
    Transformation,
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    InstanceSplitter,
    #ExpectedNumInstanceSampler,
    ExactNumInstanceSampler
)
from .deepar_network import RolledDeepARTrainingNetwork, DeepARPredictionNetwork


class DeepAREstimator(PTSEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        input_size: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None, # [Note] past data길이
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "LSTM",
        dropout_rate: float = 0.1,
        use_feat_dynamic_real: bool = False,
        use_feat_dynamic_cat: bool = False,
        use_feat_static_cat: bool = False,
        use_feat_static_real: bool = False,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        dc_cardinality: Optional[List[int]] = None,
        dc_embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,   # [Note] lags_seq : 반복되는 주기
        moving_avg_windows: Optional[List[int]] = [],   # moving average 적용할 window크기
        time_features: Optional[List[TimeFeature]] = None,
        pick_incomplete: bool = True,
        num_parallel_samples: int = 100,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__(trainer=trainer)

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.distr_output.dtype = dtype
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.dropout_rate = dropout_rate
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_dynamic_cat = use_feat_dynamic_cat
        self.use_feat_static_cat = use_feat_static_cat
        self.use_feat_static_real = use_feat_static_real
        self.cardinality = cardinality if cardinality and use_feat_static_cat else [1]
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None
            else [min(50, (cat + 1) // 2) for cat in self.cardinality]
        )
        self.dc_cardinality = dc_cardinality
        self.dc_embedding_dimension = dc_embedding_dimension
        self.scaling = scaling
        self.lags_seq = (
            lags_seq if lags_seq is not None else get_lags_for_frequency(freq_str=freq)
        )
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )
        self.moving_avg_windows = moving_avg_windows
        self.history_length = self.context_length + max(self.lags_seq) + (max(self.moving_avg_windows) if len(self.moving_avg_windows)>0 else 0)
        self.num_parallel_samples = num_parallel_samples
        self.pick_incomplete = pick_incomplete

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if not self.use_feat_static_real:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if not self.use_feat_dynamic_cat:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_CAT)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                if not self.use_feat_static_cat
                else []
            )
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
                if not self.use_feat_static_real
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT, expected_ndim=1, dtype=np.long,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL, expected_ndim=1, dtype=self.dtype,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension
                    expected_ndim=1 + len(self.distr_output.event_shape),
                    dtype=self.dtype,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=self.dtype,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True, # [Note] log scale적용
                    dtype=self.dtype,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.use_feat_dynamic_real
                        else []
                    )
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExactNumInstanceSampler(num_instances=1),
                    past_length=self.history_length,
                    future_length=self.prediction_length,
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        FieldName.ACC_TARGET_SUM,
                        FieldName.FEAT_DYNAMIC_CAT,
                        FieldName.FEAT_DYNAMIC_PAST,
                        FieldName.OBSERVED_VALUES,
                    ],                    
                    pick_incomplete=self.pick_incomplete,
                ),
            ]
        )

    def create_training_network(self, device: torch.device) -> RolledDeepARTrainingNetwork:
        return RolledDeepARTrainingNetwork(
            input_size=self.input_size,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,            
            dc_cardinality=self.dc_cardinality,
            dc_embedding_dimension=self.dc_embedding_dimension,
            lags_seq=self.lags_seq,
            moving_avg_windows=self.moving_avg_windows,
            scaling=self.scaling,
            dtype=self.dtype,
        ).to(device)

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: nn.Module,
        device: torch.device,
    ) -> Predictor:
        prediction_network = DeepARPredictionNetwork(
            num_parallel_samples=self.num_parallel_samples,
            input_size=self.input_size,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            dc_cardinality=self.dc_cardinality,
            dc_embedding_dimension=self.dc_embedding_dimension,
            lags_seq=self.lags_seq,
            moving_avg_windows=self.moving_avg_windows,
            scaling=self.scaling,
            dtype=self.dtype,
        ).to(device)

        copy_parameters(trained_network, prediction_network)

        return PTSPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
            dtype=self.dtype,
            #forecast_generator = SampleForecastGenerator() # [Note] Default는 샘플링하는 forecaster
        )
