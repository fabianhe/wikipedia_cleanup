from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm

import numpy as np

from wikipedia_cleanup.data_filter import KeepAttributesDataFilter
from wikipedia_cleanup.data_processing import get_data


class TrainAndPredictFramework:

    def __init__(self, predictor: 'Predictor', test_start_date: datetime = datetime(2018, 9, 1),
                 test_duration: int = 365):
        self.test_start_date = test_start_date
        self.test_duration = test_duration
        # total_time_window = timedelta(testset_duration)  # days
        # testset_end = testset_start + total_time_window
        # time_offset = timedelta(1)

        self.predictor = predictor
        own_relevant_attributes = ['page_id', 'value_valid_from']  # 'infobox_key', 'property_name',
        self.relevant_attributes = list(set(own_relevant_attributes + predictor.get_relevant_attributes()))
        self.data: pd.DataFrame = pd.DataFrame()

    def load_data(self, input_path: Path, n_files: int, n_jobs: int):
        filters = [KeepAttributesDataFilter(self.relevant_attributes)]
        self.data = get_data(input_path, n_files=n_files, n_jobs=n_jobs, filters=filters)
        self.data['value_valid_from'] = self.data['value_valid_from'].dt.tz_localize(None)

    def fit_model(self):
        train_data = self.data[self.data['value_valid_from'] < self.test_start_date]
        self.predictor.fit(train_data, self.test_start_date)

    def test_model(self):
        page_ids = self.data['page_id'].unique()
        # property_change_history = self.data.groupby(['page_id']).agg(list)
        all_day_labels = []

        testing_timeframes = [1, 7, 30, 365]
        timeframe_labels = ['day', 'week', 'month', 'year']
        predictions = [[] for _ in testing_timeframes]
        for page_id in tqdm(page_ids):
            # for page_id, changes in tqdm(property_change_history.iteritems(), total=len(property_change_history)):
            days_evaluated = 0
            relevant_page_ids = self.predictor.get_relevant_ids(page_id)
            current_data = self.data[self.data['page_id'].isin(relevant_page_ids)]

            # changes = np.sort(changes)
            # train_data_idx = np.searchsorted(changes, current_date, side="right")
            current_page_predictions = [[] for _ in testing_timeframes]
            test_dates = pd.date_range(self.test_start_date, self.test_start_date + timedelta(days=self.test_duration))
            day_labels = [date in self.data[self.data['page_id'] == page_id]['value_valid_from'] for date in test_dates]
            train_input = current_data[current_data['value_valid_from'] < test_dates[0]]
            for current_date in test_dates:
                # todo provide data of other page_ids for the current day.
                for i, timeframe in enumerate(testing_timeframes):
                    if days_evaluated % timeframe == 0:
                        current_page_predictions[i].append(
                            self.predictor.predict_timeframe(train_input, current_date, timeframe))
                days_evaluated += 1
            for i, prediction in enumerate(current_page_predictions):
                predictions[i].append(prediction)
            all_day_labels.append(day_labels)
        predictions = [np.array(prediction, dtype=np.bool) for prediction in predictions]

        all_day_labels = np.array(all_day_labels, dtype=np.bool)
        labels = [self.aggregate_labels(all_day_labels, timeframe) for timeframe in testing_timeframes]

        prediction_stats = []
        for y_true, y_hat, title in zip(labels, predictions, timeframe_labels):
            prediction_stats.append(self.evaluate_prediction(y_true, y_hat, title))
        return prediction_stats

    def aggregate_labels(self, labels, n):
        if n == 1:
            return labels
        if self.test_duration % n != 0:
            padded_labels = np.pad(labels, ((0, 0), (0, n - self.test_duration % n)))
        else:
            padded_labels = labels
        padded_labels = padded_labels.reshape(-1, n, labels.shape[0])
        return np.any(padded_labels, axis=1).reshape(labels.shape[0], -1)

    @staticmethod
    def print_stats(pre_rec_f1_stat, title):
        print(f"{title} \t\t changes \t no changes")
        print(f"Precision:\t\t {pre_rec_f1_stat[0][1]:.4} \t\t {pre_rec_f1_stat[0][0]:.4}")
        print(f"Recall:\t\t\t {pre_rec_f1_stat[1][1]:.4} \t\t {pre_rec_f1_stat[1][0]:.4}")
        print(f"F1score:\t\t {pre_rec_f1_stat[2][1]:.4} \t\t {pre_rec_f1_stat[2][0]:.4}")
        print(
            f"Percent of Data:\t {pre_rec_f1_stat[3][1] / (pre_rec_f1_stat[3][0] + pre_rec_f1_stat[3][1]):.4}, \tTotal: {pre_rec_f1_stat[3][1]}")
        print()

    def evaluate_prediction(self, labels: np.ndarray, prediction: np.ndarray, title: str):
        stats = precision_recall_fscore_support(labels.flatten(), prediction.flatten())
        self.print_stats(stats, title)
        return stats

    def run_pipeline(self):
        pass


def next_change(previous_change_timestamps: np.ndarray) -> Optional[datetime]:
    if len(previous_change_timestamps) < 2:
        return None

    mean_time_to_change: timedelta = np.mean(
        previous_change_timestamps[1:] - previous_change_timestamps[0:-1]
    )
    return_value: datetime = previous_change_timestamps[-1] + mean_time_to_change
    return return_value


class Predictor:
    @staticmethod
    def get_relevant_attributes():
        return []

    def fit(self, train_data: pd.DataFrame, last_day: datetime):
        pass

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

    def predict_timeframe(self, data: pd.DataFrame, current_day: datetime, timeframe: int):
        return True

    def get_relevant_ids(self, identifier):
        return [identifier]
