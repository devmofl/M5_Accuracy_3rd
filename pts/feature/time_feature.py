from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from pts.core.component import validated
from .utils import get_granularity


class TimeFeature(ABC):
    @validated()
    def __init__(self, normalized: bool = True):
        self.normalized = normalized

    @abstractmethod
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass


class MinuteOfHour(TimeFeature):
    """
    Minute of hour encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.minute / 59.0 - 0.5
        else:
            return index.minute.map(float)


class HourOfDay(TimeFeature):
    """
    Hour of day encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.hour / 23.0 - 0.5
        else:
            return index.hour.map(float)


class DayOfWeek(TimeFeature):
    """
    Hour of day encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.dayofweek / 6.0 - 0.5
        else:
            return index.dayofweek.map(float)


class DayOfMonth(TimeFeature):
    """
    Day of month encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return (index.day - 1) / 30.0 - 0.5 # day: 1~31
        else:
            return index.day.map(float)


class DayOfYear(TimeFeature):
    """
    Day of year encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return (index.dayofyear - 1) / 364.0 - 0.5 # dayofyear: 1~365
        else:
            return index.dayofyear.map(float)


class MonthOfYear(TimeFeature):
    """
    Month of year encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return (index.month - 1) / 11.0 - 0.5 # month: 1~12
        else:
            return index.month.map(float)

class Year(TimeFeature):
    """
    year encoded as value between [-0.5, 0.5]
    """    
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return (index.year - 2014) / 5.0    #TODO
        else:
            return index.year.map(float)

class WeekOfYear(TimeFeature):
    """
    Week of year encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return (index.weekofyear -1) / 51.0 - 0.5 # weekofyear: 1~52
        else:
            return index.weekofyear.map(float)


class FourierDateFeatures(TimeFeature):
    @validated()
    def __init__(self, freq: str) -> None:
        super().__init__()
        # reoccurring freq
        freqs = [
            "month",
            "day",
            "hour",
            "minute",
            "weekofyear",
            "weekday",
            "dayofweek",
            "dayofyear",
            "daysinmonth",
        ]

        assert freq in freqs
        self.freq = freq

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        values = getattr(index, self.freq)
        num_values = max(values) + 1
        steps = [x * 2.0 * np.pi / num_values for x in values]
        return np.vstack([np.cos(steps), np.sin(steps)])


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.

    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    """
    _, granularity = get_granularity(freq_str)
    if granularity == "M":
        feature_classes = [MonthOfYear]
    elif granularity == "W":
        feature_classes = [DayOfMonth, WeekOfYear]
    elif granularity in ["D", "B"]:
        feature_classes = [DayOfWeek, DayOfMonth, DayOfYear]
    elif granularity == "H":
        feature_classes = [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
    elif granularity in ["min", "T"]:
        feature_classes = [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
    else:
        supported_freq_msg = f"""
        Unsupported frequency {freq_str}

        The following frequencies are supported:

            M   - monthly
            W   - week
            D   - daily
            H   - hourly
            min - minutely
        """
        raise RuntimeError(supported_freq_msg)

    return [cls() for cls in feature_classes]


def fourier_time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    offset = to_offset(freq_str)
    granularity = offset.name

    features = {
        "M": ["weekofyear"],
        "W-SUN": ["daysinmonth", "weekofyear"],
        "W-MON": ["daysinmonth", "weekofyear"],
        "D": ["dayofweek"],
        "B": ["dayofweek", "dayofyear"],
        "H": ["hour", "dayofweek"],
        "min": ["minute", "hour", "dayofweek"],
        "T": ["minute", "hour", "dayofweek"],
    }

    assert granularity in features, f"freq {granularity} not supported"

    feature_classes: List[TimeFeature] = [
        FourierDateFeatures(freq=freq) for freq in features[granularity]
    ]
    return feature_classes
