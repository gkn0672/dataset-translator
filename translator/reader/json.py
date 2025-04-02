import json
import os
import glob
from typing import Dict, Generator, List
from datasets import IterableDataset


def read_jsonl_file(
    file_path: str, verbose: bool = False
) -> Generator[Dict, None, None]:
    """
    Read a JSONL file (each line is a valid JSON object) and yield each object.

    Args:
        file_path: Path to the JSONL file
        verbose: Whether to print verbose logs

    Yields:
        Each JSON object from the file
    """
    if verbose:
        print(f"Reading JSONL file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        line_count = 0
        error_count = 0

        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                yield json.loads(line)
                line_count += 1
            except json.JSONDecodeError:
                error_count += 1
                if verbose:
                    print(
                        f"Warning: Could not parse line {line_num} in file {file_path}. Skipping."
                    )

        if verbose:
            print(f"Read {line_count} valid JSON objects from {file_path}")
            if error_count > 0:
                print(f"Encountered {error_count} parsing errors")


def read_json_file(
    file_path: str, verbose: bool = False
) -> Generator[Dict, None, None]:
    """
    Read a JSON file that may contain an array of objects or a single object.

    Args:
        file_path: Path to the JSON file
        verbose: Whether to print verbose logs

    Yields:
        Each JSON object from the file
    """
    if verbose:
        print(f"Reading JSON file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            content = json.load(f)

            # If content is a list, yield each item
            if isinstance(content, list):
                if verbose:
                    print(f"Found array with {len(content)} items in {file_path}")
                for item in content:
                    yield item
            # If content is a dict, yield it directly
            elif isinstance(content, dict):
                if verbose:
                    print(f"Found single JSON object in {file_path}")
                yield content
            else:
                print(
                    f"Warning: Unexpected content type in {file_path}: {type(content)}"
                )

        except json.JSONDecodeError:
            if verbose:
                print(
                    f"Failed to parse {file_path} as standard JSON, trying JSONL format..."
                )

            # If standard JSON parsing fails, try JSONL format
            f.seek(0)  # Reset file pointer to start
            yield from read_jsonl_file(file_path, verbose)


def get_json_reader(
    directory_path: str,
    file_pattern: str = "*.json",
    recursive: bool = False,
    verbose: bool = False,
) -> Dict[str, IterableDataset]:
    """
    Create a dictionary of IterableDatasets for JSON files in the given directory.

    Args:
        directory_path: Path to directory containing JSON files
        file_pattern: Glob pattern to match files
        recursive: Whether to search recursively in subdirectories
        verbose: Whether to print verbose logs

    Returns:
        Dictionary with 'train' key mapping to an IterableDataset
    """
    # Check if directory exists
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Find all matching files
    if recursive:
        search_pattern = os.path.join(directory_path, "**", file_pattern)
        json_files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(directory_path, file_pattern)
        json_files = glob.glob(search_pattern)

    if not json_files:
        raise FileNotFoundError(
            f"No files matching '{file_pattern}' found in {directory_path}"
        )

    if verbose:
        print(
            f"Found {len(json_files)} files matching '{file_pattern}' in {directory_path}"
        )

    # Define generator function to yield from all files
    def combined_generator():
        for file_path in json_files:
            # Try to determine file format based on extension or content
            if file_path.lower().endswith(".jsonl"):
                yield from read_jsonl_file(file_path, verbose)
            else:
                yield from read_json_file(file_path, verbose)

    # Create and return the dataset
    return {"train": IterableDataset.from_generator(combined_generator)}


def get_dataset_from_files(
    file_paths: List[str], verbose: bool = False
) -> Dict[str, IterableDataset]:
    """
    Create a dataset from specific JSON files.

    Args:
        file_paths: List of file paths to read
        verbose: Whether to print verbose logs

    Returns:
        Dictionary with 'train' key mapping to an IterableDataset
    """
    # Check if files exist
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    if verbose:
        print(f"Reading from {len(file_paths)} specified files")

    # Define generator function to yield from all specified files
    def combined_generator():
        for file_path in file_paths:
            # Try to determine file format based on extension or content
            if file_path.lower().endswith(".jsonl"):
                yield from read_jsonl_file(file_path, verbose)
            else:
                yield from read_json_file(file_path, verbose)

    # Create and return the dataset
    return {"train": IterableDataset.from_generator(combined_generator)}
