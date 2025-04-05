import os
from typing import Dict, Optional
from datasets import IterableDataset

# Import specific readers
from translator.file.json.processor import (
    get_json_reader,
    read_json_file,
    read_jsonl_file,
)
from translator.file.parquet.processor import get_parquet_reader, read_parquet_file


def detect_file_types(directory_path: str, recursive: bool = True) -> Dict[str, int]:
    """
    Detect the types and counts of files in a directory.

    Args:
        directory_path: Directory to scan
        recursive: Whether to search recursively

    Returns:
        Dictionary with file extensions as keys and counts as values
    """
    file_types = {}

    if recursive:
        for root, _, files in os.walk(directory_path):
            for file in files:
                # Get file extension
                _, ext = os.path.splitext(file.lower())
                if ext:
                    # Remove the dot from extension
                    ext = ext[1:]
                    file_types[ext] = file_types.get(ext, 0) + 1
    else:
        for file in os.listdir(directory_path):
            if os.path.isfile(os.path.join(directory_path, file)):
                _, ext = os.path.splitext(file.lower())
                if ext:
                    ext = ext[1:]
                    file_types[ext] = file_types.get(ext, 0) + 1

    return file_types


def get_reader(
    directory_path: str,
    file_type: Optional[str] = None,
    recursive: bool = True,
    verbose: bool = False,
) -> Dict[str, IterableDataset]:
    """
    Get the appropriate reader based on file type or auto-detect.

    Args:
        directory_path: Path to directory containing files
        file_type: Optional file type override ('json', 'parquet', etc.)
        batch_size: Batch size for reading files
        recursive: Whether to search subdirectories
        verbose: Whether to print verbose logs

    Returns:
        Dictionary with 'train' key mapping to an IterableDataset
    """
    # Check if directory exists
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # If file_type is specified, use the corresponding reader
    if file_type:
        file_type = file_type.lower()
        if file_type == "json":
            if verbose:
                print(f"Using JSON reader for {directory_path}")
            return get_json_reader(directory_path, recursive=recursive, verbose=verbose)
        elif file_type == "parquet":
            if verbose:
                print(f"Using Parquet reader for {directory_path}")
            return get_parquet_reader(
                directory_path, recursive=recursive, verbose=verbose
            )
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    # Auto-detect file types if not specified
    file_types = detect_file_types(directory_path, recursive)

    if verbose:
        print(f"Detected file types: {file_types}")

    # Prioritize readers based on file counts
    if "parquet" in file_types:
        if verbose:
            print(f"Auto-detected Parquet files ({file_types['parquet']} files)")
        return get_parquet_reader(directory_path, recursive=recursive, verbose=verbose)
    elif any(ext in file_types for ext in ["json", "jsonl"]):
        if verbose:
            json_count = file_types.get("json", 0)
            jsonl_count = file_types.get("jsonl", 0)
            print(f"Auto-detected JSON files ({json_count + jsonl_count} files)")
        return get_json_reader(directory_path, recursive=recursive, verbose=verbose)
    else:
        # If no specific types are found, try JSON as a fallback (most flexible)
        if verbose:
            print("No specific file types detected, using JSON reader as fallback")
        return get_json_reader(directory_path, recursive=recursive, verbose=verbose)


def count_records(
    directory_path: str,
    file_type: Optional[str] = None,
    recursive: bool = True,
    verbose: bool = False,
) -> int:
    """
    Count the number of records across all files.

    Args:
        directory_path: Directory containing files
        file_type: Optional file type override ('json', 'parquet', etc.)
        recursive: Whether to search subdirectories
        verbose: Whether to print verbose logs

    Returns:
        Total count of records
    """
    # Check if directory exists
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # If file_type is specified, use the corresponding counter
    if file_type:
        file_type = file_type.lower()
        if file_type == "json":
            from translator.file.json.utils import count_local_records_json

            return count_local_records_json(directory_path, recursive, verbose)
        elif file_type == "parquet":
            from translator.file.parquet.utils import count_local_records_parquet

            return count_local_records_parquet(directory_path, recursive, verbose)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    # Auto-detect file types if not specified
    file_types = detect_file_types(directory_path, recursive)

    # Count records based on detected file types
    total_count = 0

    if "parquet" in file_types:
        if verbose:
            print(f"Counting records in Parquet files...")
        from translator.file.parquet.utils import count_local_records_parquet

        parquet_count = count_local_records_parquet(directory_path, recursive, verbose)
        total_count += parquet_count

    if "json" in file_types or "jsonl" in file_types:
        if verbose:
            print(f"Counting records in JSON files...")
        from translator.file.json.utils import count_local_records_json

        json_count = count_local_records_json(directory_path, recursive, verbose)
        total_count += json_count

    return total_count


def get_specific_file_reader(file_path: str, verbose: bool = False) -> Dict:
    """
    Get a reader for a specific file based on its extension.

    Args:
        file_path: Path to the file
        batch_size: Batch size for reading
        verbose: Whether to print verbose logs

    Returns:
        Dictionary with the file contents
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get file extension
    _, ext = os.path.splitext(file_path.lower())
    ext = ext[1:] if ext else ""

    if ext == "parquet":
        # For parquet files
        if verbose:
            print(f"Reading {file_path} as Parquet")

        def parquet_generator():
            yield from read_parquet_file(file_path, verbose=verbose)

        return {"train": IterableDataset.from_generator(parquet_generator)}

    elif ext in ["json", "jsonl"]:
        # For JSON files
        if verbose:
            print(f"Reading {file_path} as JSON")

        if ext == "jsonl":

            def jsonl_generator():
                yield from read_jsonl_file(file_path, verbose)

            return {"train": IterableDataset.from_generator(jsonl_generator)}
        else:

            def json_generator():
                yield from read_json_file(file_path, verbose)

            return {"train": IterableDataset.from_generator(json_generator)}

    else:
        # Try JSON as fallback for unknown extensions
        if verbose:
            print(f"Unknown file extension for {file_path}, trying as JSON")

        def json_generator():
            yield from read_json_file(file_path, verbose)

        return {"train": IterableDataset.from_generator(json_generator)}
