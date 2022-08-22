import pickle
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from wikipedia_cleanup.predictor import CachedPredictor


class PropertyCorrelationPredictor(CachedPredictor):
    def __init__(
        self,
        use_cache: bool = True,
        allowed_change_delay: int = 3,
        num_required_changes: int = 5,
        max_allowed_properties: int = 53,
        percent_allowed_mismatch: float = 0.05,
        use_num_changes: bool = False,
    ) -> None:
        super().__init__(use_cache)
        self.related_properties_lookup: dict = {}

        self.NUM_REQUIRED_CHANGES = num_required_changes
        self.MAX_ALLOWED_PROPERTIES = (
            max_allowed_properties  # Set via boxplot whisker for all links (53)
        )
        # TODO justify choice
        self.PERCENT_ALLOWED_MISMATCHES = percent_allowed_mismatch

        # TODO justify choice
        self.delay_range = allowed_change_delay
        self.use_num_changes = use_num_changes

    @staticmethod
    def _get_links(train_data: pd.DataFrame) -> Dict[str, List[str]]:
        regex_str = '\\[\\[((?:\\w+:)?[^<>\\[\\]"\\|]+)(?:\\|[^\\n\\]]+)?\\]\\]'
        regex = re.compile(regex_str)

        related_page_index = {}

        grouped_infoboxes = train_data.groupby("page_title")["current_value"].unique()

        for key, row in grouped_infoboxes.iteritems():
            related_page_index[key] = list(
                set(
                    match.groups()[0].strip()
                    for value in row
                    if value
                    for match in regex.finditer(value)
                    if not match.groups()[0].startswith(("Image:", "File:"))
                )
            )

        return related_page_index

    @staticmethod
    def _create_num_changes_time_series(
        bin_idx_and_num_changes: Any, duration: int
    ) -> csr_matrix:
        num_changes = np.array(bin_idx_and_num_changes["num_changes"])
        positions = np.array(bin_idx_and_num_changes["bin_idx"])
        series = np.zeros(duration)
        series[positions] = num_changes
        return csr_matrix(series)

    @staticmethod
    def _create_binary_time_series(a: Any, duration: int) -> csr_matrix:
        series = np.zeros(duration)
        uniques, counts = np.unique(a, return_counts=True)
        series[uniques] = counts
        return csr_matrix(series)

    def _sparse_time_series_conversion(
        self, train_data: pd.DataFrame, keys: List[str]
    ) -> pd.Series:
        bins = pd.date_range(
            train_data["value_valid_from"].min().date(),
            train_data["value_valid_from"].max().date() + timedelta(1),
        )
        total_days = len(bins)
        bins = pd.cut(train_data["value_valid_from"], bins, labels=False, right=False)
        train_data["bin_idx"] = bins

        groups = train_data.groupby(keys)
        min_support_groups = train_data[
            groups["bin_idx"].transform("count") > self.NUM_REQUIRED_CHANGES
        ].groupby(list(set(["page_title"] + keys)))

        if self.use_num_changes:
            min_support_groups = min_support_groups.apply(
                PropertyCorrelationPredictor._create_num_changes_time_series,
                duration=total_days,
            )
            min_support_groups.rename("bin_idx", inplace=True)
        else:
            min_support_groups = min_support_groups["bin_idx"].apply(
                PropertyCorrelationPredictor._create_binary_time_series,
                duration=total_days,
            )
        return min_support_groups

    def _load_cache_file(self, file_object: Any) -> bool:
        cache = pickle.load(file_object)
        if (
            cache["num_required_changes"] != self.NUM_REQUIRED_CHANGES
            or cache["max_allowed_properties"] != self.MAX_ALLOWED_PROPERTIES
            or cache["percent_allowed_mismatch"] < self.PERCENT_ALLOWED_MISMATCHES
        ):
            return False
        self.related_properties_lookup = cache["related_properties_lookup"]
        return True

    def _get_cache_object(self) -> Any:
        return {
            "num_required_changes": self.NUM_REQUIRED_CHANGES,
            "max_allowed_properties": self.MAX_ALLOWED_PROPERTIES,
            "percent_allowed_mismatch": self.PERCENT_ALLOWED_MISMATCHES,
            "related_properties_lookup": self.related_properties_lookup,
        }

    def _adjust_cache(self):
        for (
            curr_property_key,
            related_properties,
        ) in self.related_properties_lookup.items():
            filtered_related_properties_without_distances = []
            for related_property_key, related_property_distance in related_properties:
                if related_property_distance <= self.PERCENT_ALLOWED_MISMATCHES:
                    filtered_related_properties_without_distances.append(
                        related_property_key
                    )
            self.related_properties_lookup[
                curr_property_key
            ] = filtered_related_properties_without_distances

    def _fit_classifier(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ):
        def percentage_manhattan_adaptive_time_lag(
            arr1: csr_matrix, arr2: csr_matrix
        ) -> float:
            arr1 = arr1.toarray()
            arr2 = arr2.toarray()
            max_changes = arr1.sum()
            mask = np.nonzero(arr1)
            error = 0.0

            for idx in mask[1]:
                needed_num_changes = arr1[0, idx]
                for off in range(
                    -min(self.delay_range, idx),
                    min(self.delay_range + 1, arr2.shape[1] - idx),
                ):
                    used_changes = min(needed_num_changes, arr2[0, idx + off])
                    arr2[0, idx + off] -= used_changes
                    needed_num_changes -= used_changes
                error += needed_num_changes

            return error / max_changes

        def percentage_manhattan_adaptive_time_lag_symmetric(
            arr1: csr_matrix, arr2: csr_matrix
        ) -> float:
            return max(
                percentage_manhattan_adaptive_time_lag(arr1, arr2),
                percentage_manhattan_adaptive_time_lag(arr2, arr1),
            )

        # elated_page_index = self._get_links(train_data)
        min_support_groups = self._sparse_time_series_conversion(train_data, keys)

        page_title_groups = min_support_groups.reset_index()
        page_title_groups["selected_key"] = list(
            zip(*[page_title_groups[key] for key in keys])
        )
        page_title_groups = page_title_groups.groupby(["page_title"])[
            ["bin_idx", "selected_key"]
        ].agg(list)

        # links = self._find_working_links(min_support_groups, related_page_index)
        # page_to_related_pages = self._get_related_page_mapping(
        #    links, related_page_index
        # )

        matches = {}
        for row in tqdm(page_title_groups.itertuples(), total=len(page_title_groups)):
            """related_items = page_title_groups.loc[page_to_related_pages[row.Index]]
            num_samples_from_links = len(row.bin_idx)
            for related_row in related_items.itertuples():
                num_samples_from_links += len(related_row.bin_idx)
                if num_samples_from_links > self.MAX_ALLOWED_PROPERTIES:
                    break
            if num_samples_from_links <= self.MAX_ALLOWED_PROPERTIES:
                for related_row in related_items.itertuples():
                    row.bin_idx.extend(related_row.bin_idx)
                    row.selected_key.extend(related_row.selected_key)"""
            input_data = vstack(row.bin_idx)
            neigh = NearestNeighbors(
                radius=self.PERCENT_ALLOWED_MISMATCHES,
                metric=percentage_manhattan_adaptive_time_lag_symmetric,
            )
            neigh.fit(input_data)
            neighbor_distances, neighbor_indices = neigh.radius_neighbors()
            for i, (neighbors, distances) in enumerate(
                zip(neighbor_indices, neighbor_distances)
            ):
                if len(neighbors) > 0:
                    match = [
                        (row.selected_key[neighbor_idx], distance)
                        for neighbor_idx, distance in zip(neighbors, distances)
                    ]
                    matches[row.selected_key[i]] = match
        self.related_properties_lookup = matches

    @staticmethod
    def _get_related_page_mapping(links, related_page_index):
        page_to_related_pages = {}
        for page_title, related_pages in related_page_index.items():
            found_related_pages = []
            for related_page in related_pages:
                if (
                    related_page in links
                    and page_title in related_page_index[related_page]
                    and related_pages != page_title
                ):
                    found_related_pages.append(related_page)
            page_to_related_pages[page_title] = found_related_pages
        return page_to_related_pages

    @staticmethod
    def _find_working_links(min_support_groups, related_page_index):
        links = set()
        for related_pages in related_page_index.values():
            for related_page in related_pages:
                links.add(related_page)
        pandas_links = pd.Series(list(links))
        links_found = set(
            pandas_links[
                pandas_links.isin(
                    min_support_groups.reset_index("page_title")["page_title"]
                )
            ]
        )
        return links_found

    def _calculate_dependent_cache_name(self, data: pd.DataFrame) -> str:
        hash_string = (
            super()._calculate_dependent_cache_name(data)
            + f"{self.NUM_REQUIRED_CHANGES},\n"
            f"{self.MAX_ALLOWED_PROPERTIES},\n"
            f"{self.delay_range},\n"
            f"{self.use_num_changes}"
        )
        return hash_string

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        # pass in which data point we are supposed to predict

        # We can't really deal with daily predictions
        #  or timeframes that are less than the self.DELAY_RANGE

        # Check how many changes/ any changes we have inside the
        # (timeframe - delay range) area. If yes, return true, else false
        # Maybe decrease the delay range here or see how many related
        #  properties have changes here to increase precision, has to be tested
        if len(additional_data) == 0:
            return False
        col_idx = columns.index("value_valid_from")
        return additional_data[-1, col_idx] >= first_day_to_predict

    def get_relevant_attributes(self) -> List[str]:
        return [
            "page_id",
            "infobox_key",
            "page_title",
            "property_name",
            "value_valid_from",
            "current_value",
            "num_changes",
        ]

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        if identifier not in self.related_properties_lookup.keys():
            return [identifier]
        return self.related_properties_lookup[identifier]
