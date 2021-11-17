import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import libarchive.public
from tqdm.contrib.concurrent import process_map

from wikipedia_cleanup.data_filter import (
    AbstractDataFilter,
    filter_changes_with,
    generate_default_filters,
    merge_filter_stats,
    write_filter_stats_to_file,
)
from wikipedia_cleanup.data_processing import json_to_infobox_changes, read_file_sorted
from wikipedia_cleanup.schema import InfoboxChange

parser = argparse.ArgumentParser(
    description="Transform any data format to our internal format. "
    "Additionally filters and sorts the data. "
    "The filters need to be added manually by editing this file."
)
parser.add_argument(
    "--input_folder",
    type=str,
    required=True,
    help="Location of the raw 7z compressed json files.",
)
parser.add_argument(
    "--output_folder",
    type=str,
    required=True,
    help="Location of the output folder to put the pickle files.",
)
parser.add_argument(
    "--test",
    default=False,
    action="store_true",
    help="Test the script by executing it on only one file.",
)
parser.add_argument(
    "--max_workers",
    type=int,
    default=2,
    help="Max number of workers for parallelization.",
)
parser.add_argument(
    "--use_default_filters",
    default=False,
    action="store_true",
    help="Use the default filters.",
)


def calculate_output_path(changes: List[InfoboxChange], output_folder: Path) -> Path:
    page_ids = [change.page_id for change in changes]
    return output_folder.joinpath(f"{min(page_ids)}-{max(page_ids)}.pickle")


def read_7z_file_sorted(input_path: Path) -> List[InfoboxChange]:
    changes: List[InfoboxChange] = []
    with libarchive.public.file_reader(str(input_path)) as archive:
        for entry in archive:
            content_bytes = bytearray("", encoding="utf_8")
            for block in entry.get_blocks():
                content_bytes += block
            content = content_bytes.decode(encoding="utf_8")
            json_objs = content.split("\n")
            # load all changes into map
            for jsonObj in filter(lambda x: x, json_objs):
                changes.extend(json_to_infobox_changes(json.loads(jsonObj)))
            # sort changes after infobox_key, property_name, change.timestamp
    return changes


def write_custom_format(changes: List[InfoboxChange], output_folder: Path) -> None:
    with open(calculate_output_path(changes, output_folder), "wb") as out_file:
        pickle.dump(changes, out_file)


def convert_file_and_apply_filters(
    input_output_path_filters: Tuple[Path, Path, List[AbstractDataFilter]]
) -> List[AbstractDataFilter]:
    input_file, output_folder, filters = input_output_path_filters
    if input_file.suffix == ".7z":
        changes = read_7z_file_sorted(input_file)
    else:
        changes = read_file_sorted(input_file)

    changes = filter_changes_with(changes, filters)
    if len(changes) > 0:
        write_custom_format(changes, output_folder)
    return filters


if __name__ == "__main__":
    args = parser.parse_args()
    # ADD YOUR FILTERS, consider: get_default_filters
    filters = generate_default_filters() if args.use_default_filters else []
    input_files = list(Path(args.input_folder).rglob("*.7z"))
    input_files.extend(list(Path(args.input_folder).rglob("*.json")))
    input_files.extend(list(Path(args.input_folder).rglob("*.pickle")))
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.test:
        input_files = input_files[:2]
        filters = generate_default_filters()

    mapped_filters = process_map(
        convert_file_and_apply_filters,
        zip(
            input_files,
            [output_folder] * len(input_files),
            [filters] * len(input_files),
        ),
        max_workers=args.max_workers,
    )
    write_filter_stats_to_file(merge_filter_stats(mapped_filters), output_folder)
