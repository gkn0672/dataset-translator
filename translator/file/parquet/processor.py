import os
import glob
from typing import Dict, Generator, List, Optional
import pandas as pd
import pyarrow.parquet as pq
from datasets import IterableDataset


def read_parquet_file(
    file_path: str, batch_size: Optional[int] = None, verbose: bool = False
) -> Generator[Dict, None, None]:
    """
    Read a parquet file in batches and yield each row as a dictionary.

    Args:
        file_path: Path to the parquet file
        batch_size: Number of rows to read at once
        verbose: Whether to print verbose logs

    Yields:
        Each row as a dictionary
    """
    if verbose:
        print(f"Reading parquet file: {file_path}")

    # Get total number of rows and determine batch size if not provided
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows

    if verbose:
        print(f"Total rows in file: {total_rows}")

    # Auto-determine batch size if not provided
    if batch_size is None:
        from translator.file.parquet.utils import determine_optimal_parquet_batch_size

        try:
            batch_size = determine_optimal_parquet_batch_size(
                file_path, verbose=verbose
            )
            if verbose:
                print(f"Auto-determined batch size: {batch_size}")
        except Exception as e:
            # Default to a reasonable batch size on error
            batch_size = min(1000, max(10, total_rows // 10))
            if verbose:
                print(f"Error determining batch size: {e}, using default: {batch_size}")

    # Process the file in batches
    row_count = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        # Convert batch to pandas DataFrame
        df_batch = batch.to_pandas()

        # Yield each row as a dictionary
        for _, row in df_batch.iterrows():
            # Convert any non-serializable objects to strings
            row_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    row_dict[col] = None
                else:
                    row_dict[col] = val

            yield row_dict
            row_count += 1

        if verbose and row_count % 10000 == 0:
            print(f"Processed {row_count} rows so far")

    if verbose:
        print(f"Finished reading {row_count} rows from {file_path}")


def get_parquet_reader(
    directory_path: str,
    file_pattern: str = "*.parquet",
    recursive: bool = False,
    verbose: bool = False,
) -> Dict[str, IterableDataset]:
    """
    Create a dictionary of IterableDatasets for parquet files in the given directory.

    Args:
        directory_path: Path to directory containing parquet files
        file_pattern: Glob pattern to match files
        recursive: Whether to search recursively in subdirectories
        batch_size: Number of rows to read at once from each file
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
        parquet_files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(directory_path, file_pattern)
        parquet_files = glob.glob(search_pattern)

    if not parquet_files:
        raise FileNotFoundError(
            f"No files matching '{file_pattern}' found in {directory_path}"
        )

    if verbose:
        print(
            f"Found {len(parquet_files)} files matching '{file_pattern}' in {directory_path}"
        )

    # Define generator function to yield from all files
    def combined_generator():
        for file_path in parquet_files:
            yield from read_parquet_file(file_path, verbose=verbose)

    # Create and return the dataset
    return {"train": IterableDataset.from_generator(combined_generator)}


def get_dataset_from_parquet_files(
    file_paths: List[str], verbose: bool = False
) -> Dict[str, IterableDataset]:
    """
    Create a dataset from specific parquet files.

    Args:
        file_paths: List of file paths to read
        batch_size: Number of rows to read at once from each file
        verbose: Whether to print verbose logs

    Returns:
        Dictionary with 'train' key mapping to an IterableDataset
    """
    # Check if files exist
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    if verbose:
        print(f"Reading from {len(file_paths)} specified parquet files")

    # Define generator function to yield from all specified files
    def combined_generator():
        for file_path in file_paths:
            yield from read_parquet_file(file_path, verbose=verbose)

    # Create and return the dataset
    return {"train": IterableDataset.from_generator(combined_generator)}
