import itertools
import math
from bisect import bisect_left
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm

from wikipedia_cleanup.data_filter import (
    KeepAttributesDataFilter,
    OnlyUpdatesDataFilter,
)
from wikipedia_cleanup.data_processing import get_data
from wikipedia_cleanup.predictor import Predictor
from wikipedia_cleanup.property_correlation import PropertyCorrelationPredictor
from wikipedia_cleanup.utils import plot_directory


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
            set(own_relevant_attributes)
            | set(predictor.get_relevant_attributes())
            | set(self.group_key)
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
            zip(*(self.data[group_key] for group_key in self.group_key))
        )

    def fit_model(self):
        train_data = self.data[self.data["value_valid_from"] < self.test_start_date]
        self.predictor.fit(train_data.copy(), self.test_start_date, self.group_key)

    def test_model(
        self,
        randomize: bool = False,
        predict_subset: float = 1.0,
        estimate_stats: bool = False,
    ):
        keys = self.initialize_keys(randomize, predict_subset)
        all_day_labels = []

        (
            test_dates,
            test_dates_with_testing_timeframes,
        ) = self.calculate_test_date_metadata()

        predictions: List[List[List[bool]]] = [[] for _ in self.testing_timeframes]
        # it's ok to discard the time and only retain the date
        # since there is only one change per day.
        try:
            self.data["value_valid_from"] = self.data["value_valid_from"].dt.date
        except AttributeError:
            pass
        columns = self.data.columns.tolist()
        num_columns = len(columns)
        value_valid_from_column_idx = columns.index("value_valid_from")
        ten_percent_of_data = max(len(keys) // 10, 1)
        key_map = {
            key: np.array(list(group))
            for key, group in itertools.groupby(self.data.to_numpy(), lambda x: x[-1])
        }

        progress_bar_it = tqdm(keys)
        for n_processed_keys, key in enumerate(progress_bar_it):
            current_data, additional_current_data = self.select_current_data(
                key, key_map, value_valid_from_column_idx, num_columns
            )

            timestamps = current_data[:, value_valid_from_column_idx]
            additional_timestamps = additional_current_data[
                :, value_valid_from_column_idx
            ]

            current_page_predictions = self.make_prediction(
                current_data,
                timestamps,
                additional_current_data,
                additional_timestamps,
                test_dates_with_testing_timeframes,
                columns,
            )
            # save labels and predictions
            for i, prediction in enumerate(current_page_predictions):
                predictions[i].append(prediction)
            timestamps_set = set(timestamps)
            day_labels = [test_date in timestamps_set for test_date in test_dates]
            all_day_labels.append(day_labels)
            if estimate_stats:
                if n_processed_keys % ten_percent_of_data == 0:
                    stats = self.evaluate_predictions(
                        predictions, all_day_labels, [], plots=False, print_output=False
                    )
                    if stats:
                        stats_dict = {
                            "🌒🎯D_pr": stats[0][0][1],
                            "🌒📞D_rc": stats[0][1][1],
                            "🌓🎯W_pc": stats[1][0][1],
                            "🌓📞W_rc": stats[1][1][1],
                        }
                        progress_bar_it.set_postfix(stats_dict, refresh=False)

        return self.evaluate_predictions(
            predictions, all_day_labels, keys, plots=True, print_output=True
        )

    def initialize_keys(self, randomize: bool, predict_subset: float):
        keys = self.data["key"].unique()
        if randomize:
            np.random.shuffle(keys)
        if predict_subset < 1:
            print(f"Predicting only {predict_subset:.2%} percent of the data.")
            subset_idx = math.ceil(len(keys) * predict_subset)
            keys = keys[:subset_idx]
        return keys

    def calculate_test_date_metadata(
        self,
    ) -> Tuple[List[date], List[Tuple[date, List[Tuple[int, date, int]]]]]:
        test_dates = [
            (self.test_start_date + timedelta(days=x)).date()
            for x in range(self.test_duration)
        ]

        # precalculate all predict day, week, month, year entries to reuse them
        test_dates_with_testing_timeframes = []
        for days_evaluated, first_day_to_predict in enumerate(test_dates):
            curr_testing_timeframes = []
            for idx, timeframe in enumerate(self.testing_timeframes):
                if days_evaluated % timeframe == 0:
                    prediction_end_date = first_day_to_predict + timedelta(
                        days=timeframe
                    )
                    curr_testing_timeframes.append(
                        (timeframe, prediction_end_date, idx)
                    )
            test_dates_with_testing_timeframes.append(
                (first_day_to_predict, curr_testing_timeframes)
            )
        # test_dates_with_testing_timeframes has the format:
        # first_date, [timeframe, end_date, timeframe_idx]
        return test_dates, test_dates_with_testing_timeframes

    def evaluate_predictions(
        self,
        predictions: List[List[List[bool]]],
        day_labels: List[List[bool]],
        keys: List[Tuple[Any]],
        plots: bool = False,
        print_output: bool = True,
    ) -> Optional[List]:
        if np.any(day_labels):
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
            if plots:
                plot_directory().mkdir(exist_ok=True, parents=True)
                self.evaluate_metric_over_time(labels, predictions)
                self.evaluate_bucketed_predictions(labels, predictions, keys)
            return prediction_stats
        return None

    def evaluate_bucketed_predictions(self, labels, predictions, keys):
        train_data = self.data[
            self.data["value_valid_from"] < self.test_start_date.date()
        ]
        n_changes = train_data.groupby("key")["value_valid_from"].count()
        bucket_limits = [0, 5, 15, 50, 100, n_changes.max() + 1]
        buckets = list(zip(bucket_limits[:-1], bucket_limits[1:]))

        bucket_stats = []
        for low, high in buckets:
            keys_in_bucket = n_changes[(n_changes >= low) & (n_changes < high)].index
            used_indices = pd.DataFrame(keys)[0].isin(keys_in_bucket).to_numpy()
            cur_labels = [arr[used_indices] for arr in labels]
            cur_predictions = [arr[used_indices] for arr in predictions]
            for timeframe_label, timeframe_prediction in zip(
                cur_labels, cur_predictions
            ):
                bucket_stats.append(
                    self.evaluate_prediction(
                        timeframe_label, timeframe_prediction, "", False
                    )
                )

        plotting_df = pd.DataFrame(
            np.array(bucket_stats)[..., :2, 1].reshape(-1, 2),
            columns=["precision", "recall"],
        )
        plotting_df["timeframe"] = list(self.testing_timeframes * len(buckets))
        plotting_df["bucket"] = list(
            itertools.chain.from_iterable(
                ([([i] * len(self.testing_timeframes)) for i in buckets])
            )
        )
        plotting_df = (
            plotting_df.set_index(["timeframe", "bucket"])
            .sort_index()
            .reset_index()
            .set_index(["bucket", "timeframe"])
        )
        plotting_df.plot(kind="bar")
        plt.ylabel("score")
        plt.savefig(plot_directory() / "bucketed.png", bbox_inches="tight")

    def evaluate_metric_over_time(self, labels, predictions):
        for i, timeframe in enumerate(self.testing_timeframes[:-1]):
            current_labels = labels[i]
            current_predictions = predictions[i]
            stats = [
                precision_recall_fscore_support(
                    current_labels[:, i],
                    current_predictions[:, i],
                    labels=[1],
                    zero_division=0,
                )
                for i in range(current_labels.shape[1])
            ]
            prec = np.array([stat[0][0] for stat in stats])
            rec = np.array([stat[1][0] for stat in stats])
            plt.figure()
            if timeframe == 1:
                average = 5
                prec = np.mean(np.reshape(prec, (-1, average)), axis=1)
                rec = np.mean(np.reshape(rec, (-1, average)), axis=1)
                plt.xlabel(f"time in {timeframe} day(s), averaged over 5 days")
            else:
                plt.xlabel(f"time in {timeframe} day(s)")

            plt.plot(prec, label="precision")
            # trend line
            x = list(range(len(prec)))
            multidim_pol = np.polyfit(x, prec, 1)
            simple_pol = np.poly1d(multidim_pol)
            plt.plot(x, simple_pol(x), "r--", color="grey")
            plt.plot(rec, label="recall")
            plt.ylabel("score")
            plt.ylim((-0.05, 1.05))
            plt.legend()
            plt.savefig(
                plot_directory() / f"over_time_{timeframe}.png", bbox_inches="tight"
            )

    @staticmethod
    def get_data_until(
        data: np.ndarray, timestamps: np.ndarray, timestamp: date
    ) -> np.ndarray:
        if len(data) > 0:
            offset = bisect_left(timestamps, timestamp)
            return data[:offset]
        else:
            return data

    def make_prediction(
        self,
        current_data: np.ndarray,
        timestamps: np.ndarray,
        related_current_data: np.ndarray,
        additional_timestamps: np.ndarray,
        test_dates_with_testing_timeframes: List[
            Tuple[date, List[Tuple[int, date, int]]]
        ],
        columns: List[str],
    ) -> List[List[bool]]:
        current_page_predictions: List[List[bool]] = [
            [] for _ in self.testing_timeframes
        ]
        for (
            first_day_to_predict,
            curr_testing_timeframes,
        ) in test_dates_with_testing_timeframes:
            property_to_predict_data = self.get_data_until(
                current_data, timestamps, first_day_to_predict
            )
            for timeframe, prediction_end_date, idx in curr_testing_timeframes:
                related_property_to_predict_data = self.get_data_until(
                    related_current_data,
                    additional_timestamps,
                    prediction_end_date,
                )
                current_page_predictions[idx].append(
                    self.predictor.predict_timeframe(
                        property_to_predict_data,
                        related_property_to_predict_data,
                        columns,
                        first_day_to_predict,
                        timeframe,
                    )
                )
        return current_page_predictions

    def select_current_data(
        self,
        key: Tuple,
        key_map: Dict[Any, np.ndarray],
        value_valid_from_column_idx: int,
        num_columns: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        current_data = key_map[key]
        relevant_keys = self.predictor.get_relevant_ids(key).copy()

        relevant_keys = list(filter(key.__ne__, relevant_keys))
        if len(relevant_keys) != 0:
            additional_current_data_list = [
                key_map[relevant_key] for relevant_key in relevant_keys
            ]
            additional_current_data = np.concatenate(additional_current_data_list)
            additional_current_data = additional_current_data[
                additional_current_data[:, value_valid_from_column_idx].argsort()
            ]
        else:
            additional_current_data = np.empty((0, num_columns))
        return current_data, additional_current_data

    def aggregate_labels(self, labels: np.ndarray, n: int) -> np.ndarray:
        if n == 1:
            return labels
        if self.test_duration % n != 0:
            padded_labels = np.pad(labels, ((0, 0), (0, (n - self.test_duration) % n)))
        else:
            padded_labels = labels
        padded_labels = padded_labels.reshape((labels.shape[0], -1, n))
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stats = precision_recall_fscore_support(
            labels.flatten(), prediction.flatten(), zero_division=0
        )
        total_positive_predictions = np.count_nonzero(prediction)
        if print_output:
            self.print_stats(stats, total_positive_predictions, title)
        return stats

    def run_pipeline(self):
        pass


if __name__ == "__main__":
    n_files = 2
    n_jobs = 4
    input_path = Path(
        "/run/media/secret/manjaro-home/secret/mp-data/custom-format-default-filtered"
    )
    input_path = Path("../../data/custom-format-default-filtered")

    model = PropertyCorrelationPredictor()
    framework = TrainAndPredictFramework(model, ["infobox_key", "property_name"])
    # framework = TrainAndPredictFramework(model, ["page_id"])
    framework.load_data(input_path, n_files, n_jobs)
    framework.fit_model()
    framework.test_model(predict_subset=0.1)
