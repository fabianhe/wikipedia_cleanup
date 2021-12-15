from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd


class Predictor(ABC):

    # def predict_day(self, data: pd.DataFrame, current_day: datetime):
    #     return False
    #
    # def predict_week(self, data: pd.DataFrame, current_day: datetime):
    #     return False
    #
    # def predict_month(self, data: pd.DataFrame, current_day: datetime):
    #     return False
    #
    # def predict_year(self, data: pd.DataFrame, current_day: datetime):
    #     return False

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, last_day: datetime) -> None:
        raise NotImplementedError()

    @staticmethod
    def get_relevant_attributes() -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def predict_timeframe(
        self, data: pd.DataFrame, current_day: date, timeframe: int
    ) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_relevant_ids(self, identifier: str) -> List[str]:
        raise NotImplementedError()


class ZeroPredictor(Predictor):
    def fit(self, train_data: pd.DataFrame, last_day: datetime) -> None:
        pass

    def predict_timeframe(
        self, data: pd.DataFrame, current_day: date, timeframe: int
    ) -> bool:
        return False

    def get_relevant_ids(self, identifier: str) -> List[str]:
        return [identifier]

    @staticmethod
    def get_relevant_attributes() -> List[str]:
        return []


class DummyPredictor(ZeroPredictor):
    def predict_timeframe(
        self, data: pd.DataFrame, current_day: date, timeframe: int
    ) -> bool:
        pred = self.next_change(data)
        if pred is None:
            return False
        return pred - current_day <= timedelta(1)

    @staticmethod
    def next_change(time_series: pd.DataFrame) -> Optional[date]:
        previous_change_timestamps = time_series["value_valid_from"].to_numpy()
        if len(previous_change_timestamps) < 2:
            return None

        mean_time_to_change: np.timedelta64 = np.mean(
            previous_change_timestamps[1:] - previous_change_timestamps[0:-1]
        )
        return_value: np.datetime64 = (
            previous_change_timestamps[-1] + mean_time_to_change
        )
        return pd.to_datetime(return_value).date()