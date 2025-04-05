import os
import pyarrow.parquet as pq
import psutil


def count_parquet_rows(file_path: str) -> int:
    """
    Count the number of rows in a parquet file without loading all data.

    Args:
        file_path: Path to the parquet file

    Returns:
        Number of rows in the file
    """
    try:
        metadata = pq.read_metadata(file_path)
        return metadata.num_rows
    except Exception as e:
        print(f"Error reading parquet metadata: {e}")
        return 0


def count_local_records_parquet(
    directory_path: str, recursive: bool = True, verbose: bool = False
) -> int:
    """
    Count records in local parquet files efficiently.

    Args:
        directory_path: Directory containing parquet files
        recursive: Whether to search subdirectories
        verbose: Whether to print progress information

    Returns:
        int: Total number of records
    """
    total_count = 0
    parquet_files = []

    # Check if directory exists
    if not os.path.isdir(directory_path):
        if verbose:
            print(f"Directory not found: {directory_path}")
        return 0

    # Find all parquet files
    if recursive:
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, file))
    else:
        parquet_files = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith(".parquet")
            and os.path.isfile(os.path.join(directory_path, f))
        ]

    if verbose:
        print(f"Found {len(parquet_files)} parquet files")
        if len(parquet_files) > 0:
            print(f"Files: {parquet_files}")

    # Process each file
    for filepath in parquet_files:
        try:
            row_count = count_parquet_rows(filepath)
            total_count += row_count

            if verbose:
                print(f"File {filepath}: {row_count} rows")

        except Exception as e:
            if verbose:
                print(f"Error processing {filepath}: {e}")

    if verbose:
        print(f"Total records found: {total_count}")

    return total_count


def determine_optimal_parquet_batch_size(
    file_path: str,
    target_memory_percent: float = 0.1,
    min_batch_size: int = 10,
    max_batch_size: int = 10000,
    verbose: bool = False,
) -> int:
    """
    Determine an optimal batch size for reading a parquet file based on:
    1. The size of the file
    2. Available system memory
    3. Number of rows in the file

    Args:
        file_path: Path to a parquet file
        target_memory_percent: Target memory usage (percentage of available RAM)
        min_batch_size: Minimum batch size regardless of memory
        max_batch_size: Maximum batch size regardless of memory
        verbose: Whether to print verbose logs

    Returns:
        Optimal batch size for reading the parquet file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get available system memory
    available_memory = psutil.virtual_memory().available
    file_size = os.path.getsize(file_path)

    if verbose:
        print(f"Available memory: {available_memory / (1024**2):.2f} MB")
        print(f"File size: {file_size / (1024**2):.2f} MB")

    # Get file metadata
    try:
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        row_groups = parquet_file.metadata.num_row_groups

        if verbose:
            print(f"Total rows: {total_rows}")
            print(f"Row groups: {row_groups}")

        # If the file has no rows, return the minimum batch size
        if total_rows == 0:
            return min_batch_size

        # Estimate average row size
        avg_row_size = file_size / total_rows

        if verbose:
            print(f"Average row size: {avg_row_size:.2f} bytes")

        # Calculate target memory for batch
        target_memory = available_memory * target_memory_percent

        # Calculate batch size based on average row size
        calculated_batch_size = int(target_memory / avg_row_size)

        # Apply min/max constraints
        batch_size = max(min_batch_size, min(max_batch_size, calculated_batch_size))

        if verbose:
            print(f"Calculated optimal batch size: {batch_size}")

        # For very small files, just read everything at once
        if total_rows < min_batch_size:
            batch_size = int(total_rows)

        return batch_size

    except Exception as e:
        if verbose:
            print(f"Error determining batch size: {e}")
        # Fallback to a reasonable default
        return 1000


def determine_optimal_batch_size_for_directory(
    directory_path: str,
    sample_files: int = 3,
    target_memory_percent: float = 0.1,
    min_batch_size: int = 10,
    max_batch_size: int = 10000,
    verbose: bool = False,
) -> int:
    """
    Determine an optimal batch size for a directory of parquet files by
    analyzing a sample of the files.

    Args:
        directory_path: Directory containing parquet files
        sample_files: Number of files to sample
        target_memory_percent: Target memory usage (percentage of available RAM)
        min_batch_size: Minimum batch size regardless of memory
        max_batch_size: Maximum batch size regardless of memory
        verbose: Whether to print verbose logs

    Returns:
        Optimal batch size for reading the parquet files
    """
    import glob

    # Find all parquet files
    parquet_files = glob.glob(
        os.path.join(directory_path, "**", "*.parquet"), recursive=True
    )

    if not parquet_files:
        if verbose:
            print("No parquet files found in directory")
        return 1000  # Default batch size

    # Sample a subset of files
    import random

    sample_size = min(sample_files, len(parquet_files))
    sample_files = random.sample(parquet_files, sample_size)

    if verbose:
        print(f"Sampling {sample_size} files to determine batch size")

    # Get batch size for each sample file
    batch_sizes = []
    for file in sample_files:
        try:
            batch_size = determine_optimal_parquet_batch_size(
                file,
                target_memory_percent=target_memory_percent
                / sample_size,  # Divide memory among samples
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                verbose=verbose,
            )
            batch_sizes.append(batch_size)
        except Exception as e:
            if verbose:
                print(f"Error processing {file}: {e}")

    if not batch_sizes:
        return 1000  # Default batch size

    # Average the batch sizes
    avg_batch_size = int(sum(batch_sizes) / len(batch_sizes))

    # Apply min/max constraints
    batch_size = max(min_batch_size, min(max_batch_size, avg_batch_size))

    if verbose:
        print(f"Determined average optimal batch size: {batch_size}")

    return batch_size
