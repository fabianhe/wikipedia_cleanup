from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import List

from wikipedia_cleanup.schema import InfoboxChange, SparseInfoboxChange

INITIAL_STATS_VALUE = -1


class FilterStats:
    def __init__(self) -> None:
        self.initial_num_changes = INITIAL_STATS_VALUE
        self.input_num_changes = INITIAL_STATS_VALUE
        self.output_num_changes = INITIAL_STATS_VALUE

    def reset(self) -> None:
        self.initial_num_changes = INITIAL_STATS_VALUE
        self.input_num_changes = INITIAL_STATS_VALUE
        self.output_num_changes = INITIAL_STATS_VALUE

    def __str__(self) -> str:
        num_total_deletions = self.initial_num_changes - self.output_num_changes
        num_self_deletions = self.input_num_changes - self.output_num_changes
        return (
            f"Initial Number of Changes: \t {self.initial_num_changes}\n"
            f"Input Number of Changes: \t {self.input_num_changes}\n"
            f"Output Number of Changes: \t {self.output_num_changes}\n\n"
            f"Filtered Total: \t\t\t\t\t {num_total_deletions} \t "
            f"{num_total_deletions / self.initial_num_changes * 100} %\n"
            f"Filtered By current Filter: \t\t {num_self_deletions} "
            f"\t current:\t {num_self_deletions / self.input_num_changes * 100} %"
            f"\t total:\t {num_self_deletions / self.initial_num_changes * 100} %\n"
        )

    def add_stats(self, other_stats: "FilterStats") -> None:
        self.initial_num_changes += other_stats.initial_num_changes
        self.input_num_changes += other_stats.input_num_changes
        self.output_num_changes += other_stats.output_num_changes


class AbstractDataFilter(ABC):
    def __init__(self) -> None:
        self._filter_stats = FilterStats()

    def filter(
        self, changes: List[InfoboxChange], initial_num_changes: int
    ) -> List[InfoboxChange]:
        if self._filter_stats.initial_num_changes != INITIAL_STATS_VALUE:
            print(
                "WARNING: Using a filter whose stats are not reset. "
                "Thus the stats will be overwritten."
            )
        self._filter_stats.initial_num_changes = initial_num_changes
        self._filter_stats.input_num_changes = len(changes)
        filtered_changes = []
        start_idx = 0
        for end_idx in range(len(changes)):
            if (
                changes[start_idx].infobox_key != changes[end_idx].infobox_key
                or changes[start_idx].property_name != changes[end_idx].property_name
            ):
                filtered_changes.extend(
                    self._filter_for_property(changes[start_idx:end_idx])
                )
                start_idx = end_idx
        self._filter_stats.output_num_changes = len(filtered_changes)
        return filtered_changes

    @abstractmethod
    def _filter_for_property(self, changes: List[InfoboxChange]) -> List[InfoboxChange]:
        pass

    @property
    def filter_stats(self) -> FilterStats:
        return self._filter_stats

    def __str__(self) -> str:
        base_print_width = 30
        return (
            f'{"+" * base_print_width}\n'
            f"{self.__class__.__name__}\n"
            f'{"+" * base_print_width}\n' + str(self.filter_stats)
        )


class DiscardAttributesDataFilter(AbstractDataFilter):
    def _filter_for_property(self, changes: List[InfoboxChange]) -> List[InfoboxChange]:
        raise NotImplementedError("This method should never be called.")

    def filter(
        self, changes: List[InfoboxChange], initial_num_changes: int
    ) -> List[InfoboxChange]:
        self.filter_stats.initial_num_changes = initial_num_changes
        self.filter_stats.input_num_changes = initial_num_changes
        self.filter_stats.output_num_changes = initial_num_changes

        sparse_changes = []
        for change in changes:
            sparse_change: InfoboxChange = SparseInfoboxChange()  # type: ignore

            sparse_change.page_id = change.page_id
            sparse_change.property_name = change.property_name
            sparse_change.value_valid_to = change.value_valid_to
            sparse_change.value_valid_from = change.value_valid_from
            sparse_change.current_value = change.current_value
            sparse_change.previous_value = change.previous_value
            sparse_change.num_changes = change.num_changes

            sparse_change.page_title = change.page_title
            sparse_change.revision_id = change.revision_id
            sparse_change.edit_type = change.edit_type
            sparse_change.property_type = change.property_type
            sparse_change.comment = change.comment
            sparse_change.infobox_key = change.infobox_key
            sparse_change.username = change.username
            sparse_change.user_id = change.user_id
            sparse_change.position = change.position
            sparse_change.template = change.template
            sparse_change.revision_valid_to = change.revision_valid_to

            sparse_changes.append(sparse_change)
        return sparse_changes


class TestDiscardAttributesDataFilter(AbstractDataFilter):
    def _filter_for_property(self, changes: List[InfoboxChange]) -> List[InfoboxChange]:
        raise NotImplementedError("This method should never be called.")

    def filter(
        self, changes: List[InfoboxChange], initial_num_changes: int
    ) -> List[InfoboxChange]:
        self.filter_stats.initial_num_changes = initial_num_changes
        self.filter_stats.input_num_changes = initial_num_changes
        self.filter_stats.output_num_changes = initial_num_changes

        sparse_changes = []
        for change in changes:
            sparse_change: InfoboxChange = SparseInfoboxChange()  # type: ignore

            sparse_change.page_id = change.page_id
            sparse_change.property_name = change.property_name
            sparse_change.value_valid_to = change.value_valid_to
            sparse_change.value_valid_from = change.value_valid_from
            sparse_change.current_value = change.current_value
            sparse_change.previous_value = change.previous_value
            sparse_change.num_changes = change.num_changes

            sparse_change.infobox_key = change.infobox_key
            sparse_change.revision_valid_to = change.revision_valid_to

            sparse_changes.append(sparse_change)
        return sparse_changes


class MinNumChangesDataFilter(AbstractDataFilter):
    _min_number_of_changes: int

    def __init__(self, min_number_of_changes: int = 5):
        super().__init__()
        self._min_number_of_changes = min_number_of_changes

    @property
    def min_number_of_changes(self) -> int:
        return self._min_number_of_changes

    def _filter_for_property(self, changes: List[InfoboxChange]) -> List[InfoboxChange]:
        return [] if len(changes) < self._min_number_of_changes else changes


class MajorityValuePerDayDataFilter(AbstractDataFilter):
    """
    This filter needs to process each tuple (infobox, property_name, day).
    Therefore the method is overwritten / slightly changes.
    """

    def filter(
        self, changes: List[InfoboxChange], initial_num_changes: int
    ) -> List[InfoboxChange]:
        self._filter_stats.initial_num_changes = initial_num_changes
        self._filter_stats.input_num_changes = len(changes)
        filtered_changes = []
        start_idx = 0
        for end_idx in range(len(changes)):
            if (
                changes[start_idx].value_valid_from.date()
                != changes[end_idx].value_valid_from.date()
                or changes[start_idx].infobox_key != changes[end_idx].infobox_key
                or changes[start_idx].property_name != changes[end_idx].property_name
            ):
                filtered_changes.extend(
                    self._filter_for_property(changes[start_idx:end_idx])
                )
                start_idx = end_idx
        self._filter_stats.output_num_changes = len(filtered_changes)
        return filtered_changes

    def _filter_for_property(self, changes: List[InfoboxChange]) -> List[InfoboxChange]:
        if len(changes) == 1:
            return [changes[0]]
        values_to_occurrences = Counter([change.current_value for change in changes])
        max_occurrence = max(
            values_to_occurrences.items(), key=lambda val_occ: val_occ[1]
        )[1]
        representative_change = deepcopy(
            next(
                filter(
                    lambda change: values_to_occurrences[change.current_value]
                    >= max_occurrence,
                    reversed(changes),
                )
            )
        )
        representative_change.value_valid_to = deepcopy(changes[-1].value_valid_to)
        representative_change.value_valid_from = deepcopy(changes[0].value_valid_from)
        representative_change.num_changes = len(changes)
        return [representative_change]


class AbstractRevertsDataFilter(AbstractDataFilter, ABC):
    @abstractmethod
    def change_pair_needs_to_be_filtered(
        self, change_a: InfoboxChange, change_b: InfoboxChange
    ) -> bool:
        pass

    def _filter_for_property(self, changes: List[InfoboxChange]) -> List[InfoboxChange]:
        filtered_changes: List[InfoboxChange] = []
        idx = 0
        while idx < len(changes) - 1:
            if self.change_pair_needs_to_be_filtered(
                changes[idx], changes[idx + 1]
            ) or self.change_pair_needs_to_be_filtered(changes[idx + 1], changes[idx]):
                if len(filtered_changes) > 0:
                    dates = [
                        date
                        for date in [
                            changes[idx].value_valid_to,
                            changes[idx].value_valid_to,
                        ]
                        if date
                    ]
                    filtered_changes[-1].value_valid_to = (
                        max(dates) if len(dates) > 0 else None
                    )
                idx += 2
            else:
                filtered_changes.append(changes[idx])
                idx += 1
        if idx < len(changes):
            filtered_changes.append(changes[-1])
        return filtered_changes


class BotRevertsDataFilter(AbstractRevertsDataFilter):
    def change_pair_needs_to_be_filtered(
        self, change_a: InfoboxChange, change_b: InfoboxChange
    ) -> bool:
        return (
            change_a.current_value == change_b.previous_value
            and change_a.previous_value == change_b.current_value
            and change_a.value_valid_to == change_b.value_valid_from
            and change_a.value_valid_from == change_a.value_valid_to
        )


class EditWarRevertsDataFilter(AbstractRevertsDataFilter):
    def __init__(self, max_time_to_reverting_change: timedelta = timedelta(days=7)):
        super().__init__()
        self.max_time_to_reverting_change = max_time_to_reverting_change

    def change_pair_needs_to_be_filtered(
        self, change_a: InfoboxChange, change_b: InfoboxChange
    ) -> bool:
        return (
            change_a.current_value == change_b.previous_value
            and change_a.previous_value == change_b.current_value
            and change_a.value_valid_to == change_b.value_valid_from
            and (change_a.value_valid_to - change_a.value_valid_from)
            <= self.max_time_to_reverting_change
        )


def generate_default_filters() -> List[AbstractDataFilter]:
    return [
        BotRevertsDataFilter(),
        MajorityValuePerDayDataFilter(),
        MinNumChangesDataFilter(),
    ]


def filter_changes_with(
    changes: List[InfoboxChange], filters: List[AbstractDataFilter]
) -> List[InfoboxChange]:
    if len(filters) == 0:
        return changes
    initial_num_changes = len(changes)
    for data_filter in filters:
        changes = data_filter.filter(changes, initial_num_changes)
    return changes


def merge_filter_stats_into(
    list_of_filters: List[List[AbstractDataFilter]],
    target_filters: List[AbstractDataFilter],
) -> None:
    if len(list_of_filters) == 0:
        return
    for filters in list_of_filters[1:]:
        for idx, filter in enumerate(filters):
            filter_name = filter.__class__.__name__
            if filter_name != target_filters[idx].__class__.__name__:
                raise ValueError("Expected all filters to have the same order.")
            target_filters[idx].filter_stats.add_stats(filter.filter_stats)


def get_stats_from_filters(filters: List[AbstractDataFilter]) -> str:
    if len(filters) == 0:
        return ""
    result = ""
    initial_num_changes = filters[0].filter_stats.initial_num_changes
    if any(
        [
            data_filter.filter_stats.initial_num_changes != initial_num_changes
            for data_filter in filters
        ]
    ):
        result += (
            "WARNING: Initial number of changes mismatch for the given filters. "
            "Filters were probably not used in the same context.\n\n"
        )
    result += "\n".join([str(data_filter) for data_filter in filters])
    return result


def write_filter_stats_to_file(
    filters: List[AbstractDataFilter], output_folder: Path
) -> None:
    with open(output_folder.joinpath("filter-stats.txt"), "wt") as out_file:
        out_file.write(get_stats_from_filters(filters))
