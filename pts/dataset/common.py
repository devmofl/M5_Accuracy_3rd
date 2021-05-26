from typing import Any, Dict, Iterable, NamedTuple, List, Optional

import pandas as pd
from pydantic import BaseModel

# Dictionary used for data flowing through the transformations.
DataEntry = Dict[str, Any]

# A Dataset is an iterable of DataEntry.
Dataset = Iterable[DataEntry]


class SourceContext(NamedTuple):
    source: str
    row: int


class FieldName:
    """
    A bundle of default field names to be used by clients when instantiating
    transformer instances.
    """

    ITEM_ID = "item_id"

    START = "start"
    TARGET = "target"
    ACC_TARGET_SUM = "accumulated_target"

    FEAT_STATIC_CAT = "feat_static_cat"
    FEAT_STATIC_REAL = "feat_static_real"
    FEAT_DYNAMIC_CAT = "feat_dynamic_cat"
    FEAT_DYNAMIC_REAL = "feat_dynamic_real"
    FEAT_DYNAMIC_PAST = "feat_dynamic_past" # 과거만 참조가능한 feature (미래정보는 없음)

    FEAT_TIME = "time_feat"
    FEAT_CONST = "feat_dynamic_const"
    FEAT_AGE = "feat_dynamic_age"

    OBSERVED_VALUES = "observed_values"
    IS_PAD = "is_pad"
    FORECAST_START = "forecast_start"


class CategoricalFeatureInfo(BaseModel):
    name: str
    cardinality: str


class BasicFeatureInfo(BaseModel):
    name: str


class MetaData(BaseModel):
    freq: str = None
    target: Optional[BasicFeatureInfo] = None

    feat_static_cat: List[CategoricalFeatureInfo] = []
    feat_static_real: List[BasicFeatureInfo] = []
    feat_dynamic_real: List[BasicFeatureInfo] = []
    feat_dynamic_cat: List[CategoricalFeatureInfo] = []

    prediction_length: Optional[int] = None


class TrainDatasets(NamedTuple):
    """
    A dataset containing two subsets, one to be used for training purposes,
    and the other for testing purposes, as well as metadata.
    """

    metadata: MetaData
    train: Dataset
    test: Optional[Dataset] = None


class DateConstants:
    """
    Default constants for specific dates.
    """

    OLDEST_SUPPORTED_TIMESTAMP = pd.Timestamp(1800, 1, 1, 12)
    LATEST_SUPPORTED_TIMESTAMP = pd.Timestamp(2200, 1, 1, 12)
