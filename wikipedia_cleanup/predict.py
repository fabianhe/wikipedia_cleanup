import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm

from wikipedia_cleanup.data_filter import (
    KeepAttributesDataFilter,
    OnlyUpdatesDataFilter,
)
from wikipedia_cleanup.data_processing import get_data
from wikipedia_cleanup.predictor import Predictor
from wikipedia_cleanup.property_correlation import PropertyCorrelationPredictor


class TrainAndPredictFramework:
    def __init__(
            self,
            predictor: Predictor,
            group_key: List[str],
            test_start_date: datetime = datetime(2018, 9, 1),
            test_duration: int = 365,
    ):
        self.test_start_date = test_start_date
        self.test_duration = test_duration
        self.group_key = group_key
        self.testing_timeframes = [1, 7, 30, 365]
        self.timeframe_labels = ["day", "week", "month", "year"]

        self.predictor = predictor
        own_relevant_attributes = ["value_valid_from"]
        self.relevant_attributes = list(
            set(
                own_relevant_attributes
                + predictor.get_relevant_attributes()
                + self.group_key
            )
        )
        self.data: pd.DataFrame = pd.DataFrame()

    def load_data(self, input_path: Path, n_files: int, n_jobs: int):
        filters = [
            OnlyUpdatesDataFilter(),
            KeepAttributesDataFilter(self.relevant_attributes),
        ]
        self.data = get_data(
            input_path, n_files=n_files, n_jobs=n_jobs, filters=filters  # type: ignore
        )
        self.data["value_valid_from"] = self.data["value_valid_from"].dt.tz_localize(
            None
        )
        self.data["key"] = list(
            zip(*[self.data[group_key] for group_key in self.group_key])
        )

    def fit_model(self):
        train_data = self.data[self.data["value_valid_from"] < self.test_start_date]
        self.predictor.fit(train_data.copy(), self.test_start_date, self.group_key)

    def test_model(self, randomize: bool = False, predict_subset: float = 1.0):
        keys = self.data["key"].unique()
        if randomize:
            np.random.shuffle(keys)
        if predict_subset < 1:
            print(f"Predicting only {predict_subset:.2%} percent of the data.")
            subset_idx = math.ceil(len(keys) * predict_subset)
            keys = keys[:subset_idx]

        test_dates = [
            (self.test_start_date + timedelta(days=x)).date()
            for x in range(self.test_duration)
        ]
        predictions: List[List[List[bool]]] = [[] for _ in self.testing_timeframes]

        self.predict_all_keys(keys, predictions, test_dates)

    def predict_all_keys(self, keys: np.ndarray, predictions: List[List[List[bool]]],
                         test_dates: List[date]):
        all_day_labels = []
        single_percent_of_data = len(keys) // 100
        progress_bar_it = tqdm(keys)
        for n_processed_keys, key in enumerate(progress_bar_it):
            current_data, additional_current_data = self.select_current_data(key)

            timestamps = self.convert_timestamps(current_data)
            additional_timestamps = self.convert_timestamps(additional_current_data)

            current_page_predictions = self.make_prediction(
                current_data,
                timestamps,
                additional_current_data,
                additional_timestamps,
                test_dates,
            )
            # save labels and predictions
            for i, prediction in enumerate(current_page_predictions):
                predictions[i].append(prediction)
            day_labels = [date in timestamps for date in test_dates]
            all_day_labels.append(day_labels)
            if n_processed_keys % single_percent_of_data == 0 and n_processed_keys > 0:
                stats = self.evaluate_predictions(predictions, all_day_labels, False)
                stats_dict = {
                    "🌒🎯D_pr": stats[0][0][1],
                    "🌒📞D_rc": stats[0][1][1],
                    "🌓🎯W_pc": stats[1][0][1],
                    "🌓📞W_rc": stats[1][1][1],
                }
                progress_bar_it.set_postfix(stats_dict, refresh=False)
        self.evaluate_predictions(predictions, all_day_labels)

    @staticmethod
    def convert_timestamps(data: pd.DataFrame) -> np.ndarray:
        return data["value_valid_from"].dt.date.to_numpy()

    def evaluate_predictions(
            self,
            predictions: List[List[List[bool]]],
            day_labels: List[List[bool]],
            print_output: bool = True,
    ) -> List:
        predictions = [
            np.array(prediction, dtype=np.bool) for prediction in predictions
        ]
        all_day_labels = np.array(day_labels, dtype=np.bool)
        labels = [
            self.aggregate_labels(all_day_labels, timeframe)
            for timeframe in self.testing_timeframes
        ]

        prediction_stats = []
        for y_true, y_hat, title in zip(labels, predictions, self.timeframe_labels):
            prediction_stats.append(
                self.evaluate_prediction(y_true, y_hat, title, print_output)
            )
        return prediction_stats

    @staticmethod
    def get_data_until(
            data: pd.DataFrame, timestamps: np.ndarray, timestamp: date
    ) -> pd.DataFrame:
        if len(data) > 0:
            offset = np.searchsorted(
                timestamps,
                timestamp,
                side="left",
            )
            return data.iloc[:offset]
        else:
            return data

    def make_prediction(
            self,
            current_data: pd.DataFrame,
            timestamps: np.ndarray,
            additional_current_data: pd.DataFrame,
            additional_timestamps: np.ndarray,
            test_dates: List[date],
    ) -> List[List[bool]]:
        current_page_predictions: List[List[bool]] = [
            [] for _ in self.testing_timeframes
        ]

        for days_evaluated, current_date in enumerate(test_dates):
            train_input = self.get_data_until(current_data, timestamps, current_date)
            for i, timeframe in enumerate(self.testing_timeframes):
                if days_evaluated % timeframe == 0:
                    additional_train_input = self.get_data_until(
                        additional_current_data,
                        additional_timestamps,
                        current_date + timedelta(days=timeframe),
                    )

                    current_page_predictions[i].append(
                        self.predictor.predict_timeframe(
                            train_input, additional_train_input, current_date, timeframe
                        )
                    )
        return current_page_predictions

    def select_current_data(self, key: Tuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
        relevant_keys = self.predictor.get_relevant_ids(key).copy()

        current_data = self.data[self.data["key"] == key].sort_values(
            by=["value_valid_from"]
        )
        relevant_keys = list(filter(key.__ne__, relevant_keys))
        additional_current_data = self.data[
            self.data["key"].isin(relevant_keys)
        ].sort_values(by=["value_valid_from"])
        return current_data, additional_current_data

    def aggregate_labels(self, labels: np.ndarray, n: int):
        if n == 1:
            return labels
        if self.test_duration % n != 0:
            padded_labels = np.pad(labels, ((0, 0), (0, (n - self.test_duration) % n)))
        else:
            padded_labels = labels
        padded_labels = padded_labels.reshape(labels.shape[0], -1, n)
        return np.any(padded_labels, axis=2)

    @staticmethod
    def print_stats(
            pre_rec_f1_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            num_pos_predictions: int,
            title: str,
    ):
        percent_data = pre_rec_f1_stat[3][1] / (
                pre_rec_f1_stat[3][0] + pre_rec_f1_stat[3][1]
        )
        percent_changes_pred = num_pos_predictions / (
                pre_rec_f1_stat[3][0] + pre_rec_f1_stat[3][1]
        )
        print(f"{title} \t\t\tchanges \tno changes")
        print(
            f"Precision:\t\t{pre_rec_f1_stat[0][1]:.4} \t\t{pre_rec_f1_stat[0][0]:.4}"
        )
        print(f"Recall:\t\t\t{pre_rec_f1_stat[1][1]:.4} \t\t{pre_rec_f1_stat[1][0]:.4}")
        print(f"F1score:\t\t{pre_rec_f1_stat[2][1]:.4} \t\t{pre_rec_f1_stat[2][0]:.4}")
        print(f"Changes of Data:\t{percent_data:.4%}, \tTotal: {pre_rec_f1_stat[3][1]}")
        print(
            f"Changes of Pred:\t{percent_changes_pred:.4%},"
            f" \tTotal: {num_pos_predictions}"
        )
        print()

    def evaluate_prediction(
            self, labels: np.ndarray, prediction: np.ndarray, title: str, print_output: bool
    ):
        stats = precision_recall_fscore_support(labels.flatten(), prediction.flatten())
        total_positive_predictions = np.count_nonzero(prediction)
        if print_output:
            self.print_stats(stats, total_positive_predictions, title)
        return stats

    def run_pipeline(self):
        pass


if __name__ == "__main__":
    n_files = 4
    n_jobs = 4
    input_path = Path("../../data/custom-format-default-filtered/")

    model = PropertyCorrelationPredictor()
    framework = TrainAndPredictFramework(model, ["infobox_key", "property_name"])
    # framework = TrainAndPredictFramework(model, ["page_id"])
    framework.load_data(input_path, n_files, n_jobs)
    framework.fit_model()
    framework.test_model()
