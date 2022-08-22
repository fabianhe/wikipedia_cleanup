import itertools
import math
import pickle
import time
from bisect import bisect_left
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wikipedia_cleanup.data_filter import (
    AbstractDataFilter,
    KeepAttributesDataFilter,
    StaticInfoboxTemplateDataAdder,
)
from wikipedia_cleanup.data_processing import get_data
from wikipedia_cleanup.evaluation import (
    ALL_EVAL_METHODS,
    create_prediction_output,
    evaluate_prediction,
)
from wikipedia_cleanup.predictor import Predictor
from wikipedia_cleanup.property_correlation import PropertyCorrelationPredictor
from wikipedia_cleanup.random_forest import RandomForestPredictor
from wikipedia_cleanup.utils import plot_directory, result_directory


class TrainAndPredictFramework:
    TEST_DURATION = 365

    TEST_SET_START_DATE = datetime(2018, 9, 1)
    VALIDATION_SET_START_DATE = TEST_SET_START_DATE - timedelta(days=TEST_DURATION)

    def __init__(
        self,
        predictor: Predictor,
        group_key: List[str],
        test_start_date: datetime = VALIDATION_SET_START_DATE,
        test_duration: int = TEST_DURATION,
        run_id: Optional[str] = None,
    ):
        self.test_start_date = test_start_date
        self.test_duration = test_duration
        self.group_key = group_key
        self.testing_timeframes = [1, 7, 30, 365]
        self.timeframe_labels = ["day", "week", "month", "year"]

        self.predictor = predictor
        own_relevant_attributes = ["value_valid_from", "template"]
        self.relevant_attributes = list(
            set(own_relevant_attributes)
            | set(predictor.get_relevant_attributes())
            | set(self.group_key)
        )
        self.data: pd.DataFrame = pd.DataFrame()
        self.run_results: Dict[str, Any] = {}
        current_time = datetime.now()
        self.run_id = (
            run_id
            if run_id is not None
            else (
                f"{current_time.date()}:{current_time.hour}-"
                f"{current_time.minute}-{current_time.second}"
            )
        )

    def load_data(
        self,
        input_path: Path,
        n_files: int,
        n_jobs: int,
        appended_filters: List[AbstractDataFilter] = None,
        static_attribute_path: Optional[Path] = None,
    ):
        filters: List[AbstractDataFilter] = []
        if appended_filters is not None:
            print(
                f"WARNING: Using additional non standard "
                f"preceding filters for the data loading: {appended_filters}"
            )
            filters.extend(appended_filters)
        filters.append(KeepAttributesDataFilter(self.relevant_attributes))
        if static_attribute_path:
            filters += [StaticInfoboxTemplateDataAdder(static_attribute_path)]

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
        print("Start training.")
        start = time.time()
        train_data = self.data[self.data["value_valid_from"] < self.test_start_date]
        self.predictor.fit(train_data.copy(), self.test_start_date, self.group_key)
        end = time.time()
        print(f"Finished training. Time elapsed: {timedelta(seconds=end - start)}")

    def test_model(
        self,
        randomize: bool = False,
        predict_subset: float = 1.0,
        save_results: bool = False,
        generate_summary: bool = True,
    ) -> Optional[str]:
        keys = self._initialize_keys(randomize, predict_subset)
        all_day_labels = []

        (
            test_dates,
            test_dates_with_testing_timeframes,
        ) = self._calculate_test_date_metadata()

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
        key_column_idx = columns.index("key")
        key_map = {
            key: np.array(list(group))
            for key, group in itertools.groupby(
                self.data.to_numpy(), lambda x: x[key_column_idx]
            )
        }

        progress_bar_it = tqdm(keys)
        for n_processed_keys, key in enumerate(progress_bar_it):
            current_data, additional_current_data = self._select_current_data(
                key, key_map, value_valid_from_column_idx, num_columns
            )

            timestamps = current_data[:, value_valid_from_column_idx]
            additional_timestamps = additional_current_data[
                :, value_valid_from_column_idx
            ]

            current_page_predictions = self._make_prediction(
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
        self._reformat_preds_and_labels(predictions, all_day_labels)
        run_statistics = None
        if np.any(all_day_labels):
            if generate_summary:
                run_statistics = self._evaluate_predictions()
        else:
            print("Results could not be generated. No changes in the test timeframe.")
        self.run_results["keys"] = keys

        if run_statistics or save_results:
            output_folder = result_directory(self.run_id)
            output_folder.mkdir(parents=True, exist_ok=True)
        if run_statistics:
            self._save_run_stats(output_folder, run_statistics)
        if save_results:
            self._save_run_results(output_folder)
        return run_statistics

    def generate_plots(self, run_results: Optional[dict] = None) -> None:
        print("Starting generating plots.")
        start = time.time()

        if not run_results:
            run_results = self.run_results
        labels = run_results["labels"]
        predictions = run_results["predictions"]
        keys = run_results["keys"]
        output_folder = plot_directory(self.run_id)
        output_folder.mkdir(parents=True, exist_ok=True)

        train_data = self.data[
            self.data["value_valid_from"] < self.test_start_date.date()
        ]

        evaluation_methods = ALL_EVAL_METHODS
        for evaluation_method in evaluation_methods:
            try:
                evaluation_method(  # type: ignore
                    labels,
                    predictions,
                    self.testing_timeframes,
                    output_folder,
                    keys,
                    train_data,
                )
            except Exception:
                print(f"{evaluation_method.__name__} failed.")

        end = time.time()
        print(f"Finished evaluation. Time elapsed: {timedelta(seconds=end - start)}")

    def _initialize_keys(self, randomize: bool, predict_subset: float):
        keys = self.data["key"].unique()
        if randomize:
            np.random.shuffle(keys)
        if predict_subset < 1:
            print(f"Predicting only {predict_subset:.2%} percent of the data.")
            subset_idx = math.ceil(len(keys) * predict_subset)
            keys = keys[:subset_idx]
        return keys

    def _calculate_test_date_metadata(
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
                if (
                    days_evaluated % timeframe == 0
                    and days_evaluated + timeframe <= self.test_duration
                ):
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

    def _evaluate_predictions(
        self,
    ) -> str:
        prediction_output = ""
        print("Starting evaluation.")
        start = time.time()

        prediction_stats = []
        pred_stats = []
        # needs to be commented out if working with probabilities as predictions
        for y_true, y_hat, title in zip(
            self.run_results["labels"],
            self.run_results["predictions"],
            self.timeframe_labels,
        ):
            pred_stats.append(
                {
                    "prec_recall": evaluate_prediction(y_true, y_hat),
                    "y_hat": y_hat,
                    "y_true": y_true,
                }
            )
            prediction_stats.append(create_prediction_output(y_true, y_hat, title))

        prediction_output = "\n\n".join(prediction_stats)

        self.pred_stats = pred_stats

        end = time.time()
        print(f"Finished evaluation. Time elapsed: {timedelta(seconds=end - start)}")

        return prediction_output

    def _reformat_preds_and_labels(
        self, predictions: List[List[List[bool]]], day_labels: List[List[bool]]
    ):

        for pred in predictions[0][0]:
            if type(pred) == float:
                predictions = [
                    np.array(prediction, dtype=float) for prediction in predictions
                ]
                break
            elif type(pred) == bool:
                predictions = [
                    np.array(prediction, dtype=bool) for prediction in predictions
                ]
                break
        all_day_labels = np.array(day_labels, dtype=np.bool)
        labels = [
            self._aggregate_labels(all_day_labels, timeframe)
            for timeframe in self.testing_timeframes
        ]
        self.run_results["labels"] = labels
        self.run_results["predictions"] = predictions

    @staticmethod
    def _get_data_until(
        data: np.ndarray, timestamps: np.ndarray, timestamp: date
    ) -> np.ndarray:
        if len(data) > 0:
            offset = bisect_left(timestamps, timestamp)
            return data[:offset]
        else:
            return data

    def _make_prediction(
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
            property_to_predict_data = self._get_data_until(
                current_data, timestamps, first_day_to_predict
            )
            for timeframe, prediction_end_date, idx in curr_testing_timeframes:
                related_property_to_predict_data = self._get_data_until(
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

    def _select_current_data(
        self,
        key: Tuple,
        key_map: Dict[Any, np.ndarray],
        value_valid_from_column_idx: int,
        num_columns: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        current_data = key_map[key]
        relevant_keys = self.predictor.get_relevant_ids(key).copy()
        relevant_keys = [
            key for key in filter(key.__ne__, relevant_keys) if key in key_map
        ]
        if len(relevant_keys) != 0:
            additional_current_data_list = [
                key_map[relevant_key] for relevant_key in relevant_keys
            ]
            additional_current_data = np.concatenate(additional_current_data_list)
            additional_current_data = additional_current_data[
                additional_current_data[:, value_valid_from_column_idx].argsort()
            ]
            return current_data, additional_current_data
        return current_data, np.empty((0, num_columns))

    def _aggregate_labels(self, labels: np.ndarray, n: int) -> np.ndarray:
        if n == 1:
            return labels
        if self.test_duration % n != 0:
            cut_labels = labels[:, : -(self.test_duration % n)]
        else:
            cut_labels = labels
        cut_labels = cut_labels.reshape((labels.shape[0], -1, n))
        return np.any(cut_labels, axis=2)

    @staticmethod
    def _save_run_stats(output_folder: Path, run_statistics: str) -> None:
        run_stats_path = output_folder / "stats.txt"
        with open(run_stats_path, "w") as f:
            f.write(run_statistics)

    def _save_run_results(self, output_folder: Path):
        run_results_path = output_folder / "results.pickle"
        with open(run_results_path, "wb") as f:
            pickle.dump(self.run_results, f)


if __name__ == "__main__":
    n_files = 2
    n_jobs = 4
    input_path = Path(
        "/run/media/secret/manjaro-home/secret/mp-data/new_costum_filtered_format"
    )
    # input_path = Path("../../data/custom-format-default-filtered")

    model1 = PropertyCorrelationPredictor()
    model2 = RandomForestPredictor()
    framework = TrainAndPredictFramework(model1, ["infobox_key", "property_name"])
    # framework = TrainAndPredictFramework(model, ["page_id"])
    framework.load_data(input_path, n_files, n_jobs)
    framework.fit_model()
    framework.test_model(predict_subset=0.1)
    framework.generate_plots()
