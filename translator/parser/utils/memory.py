import psutil
from typing import List, Dict
import json
import sys


def estimate_item_memory_usage(sample_items: List[Dict]) -> float:
    """
    Estimate memory usage per item based on a sample.

    Args:
        sample_items: A list of sample items to estimate memory usage

    Returns:
        Estimated memory usage per item in bytes
    """
    if not sample_items:
        return 1000  # Default estimate if no samples

    # Estimate memory usage based on serialized size
    total_size = 0
    for item in sample_items:
        # Serialize the item to get a rough estimate of its size
        serialized = json.dumps(item)
        total_size += sys.getsizeof(serialized)

        # Also account for the dictionary overhead
        total_size += sys.getsizeof(item)

        # Add estimates for each field
        for key, value in item.items():
            total_size += sys.getsizeof(key)
            total_size += sys.getsizeof(value)

    # Return average size per item with a safety margin
    return (total_size / len(sample_items)) * 1.5  # Add 50% safety margin


def determine_optimal_batch_size(
    item_memory_estimate: float,
    max_memory_percent: float = 0.2,
    min_batch_size: int = 10,
    max_batch_size: int = 5000,
) -> int:
    """
    Determine the optimal batch size based on available memory and sample data.

    Args:
        item_memory_estimate: Estimated memory per item in bytes
        max_memory_percent: Target memory usage (percentage of available RAM)
        min_batch_size: Minimum batch size regardless of memory
        max_batch_size: Maximum batch size regardless of memory

    Returns:
        Optimal batch size
    """
    # Get available system memory
    available_memory = psutil.virtual_memory().available
    target_memory = available_memory * max_memory_percent

    if item_memory_estimate is None or item_memory_estimate <= 0:
        # If we don't have an estimate, use a default batch size
        return 100

    # Calculate batch size based on memory target
    optimal_batch_size = int(target_memory / item_memory_estimate)

    # Apply min/max constraints
    optimal_batch_size = max(min_batch_size, min(max_batch_size, optimal_batch_size))

    return optimal_batch_size


def sample_dataset(data_generator, n_samples: int = 20) -> List[Dict]:
    """
    Sample the dataset to estimate memory usage.

    Args:
        data_generator: Generator yielding dataset items
        n_samples: Number of samples to collect

    Returns:
        List of sample items
    """
    samples = []

    # Collect samples
    for _ in range(n_samples):
        try:
            item = next(data_generator)
            samples.append(item)
        except StopIteration:
            break

    return samples
