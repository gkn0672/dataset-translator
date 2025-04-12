import sys
import time
import socket
import requests

sys.path.insert(0, r"./")
from functools import wraps
from tqdm.auto import tqdm

import tempfile
import os
import shutil
from datasets import load_dataset, get_dataset_config_names, load_dataset_builder
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def safe_tqdm_write(text_to_write: str) -> None:
    """
    Writes the given text to the tqdm progress bar if it exists, otherwise prints it.

    Args:
        text_to_write (str): The text to be written.

    Returns:
        None
    """
    try:
        if text_to_write:
            if hasattr(tqdm, "_instances"):
                tqdm.write(text_to_write)
            else:
                print(text_to_write)
    except Exception as e:
        print(f"Error in safe_tqdm_write: {e}")
        print(text_to_write)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")

        return result

    return timeit_wrapper


def have_internet(host="8.8.8.8", port=53, timeout=3) -> bool:
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        print(ex)
        return False


def get_dataset_info(dataset_name, api_token=None):
    """
    Retrieve dataset information from Hugging Face datasets-server.
    First tries with 'en' config, then falls back to 'default' config if needed.

    Parameters:
    -----------
    dataset_name : str
        The name of the dataset (e.g., 'matteogabburo/mASNQ')
    api_token : str, optional
        Hugging Face API token for accessing private datasets

    Returns:
    --------
    dict
        Dataset information including total num_examples across configs
    """
    # Set up headers with authorization token if provided
    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    # First try with 'en' config
    en_api_url = (
        f"https://datasets-server.huggingface.co/info?dataset={dataset_name}&config=en"
    )

    response = requests.get(en_api_url, headers=headers)

    # If 'en' config fails, try 'default' config
    if response.status_code == 404:
        default_api_url = f"https://datasets-server.huggingface.co/info?dataset={dataset_name}&config=default"
        response = requests.get(default_api_url, headers=headers)

        if response.status_code != 200:
            error_msg = f"Error: {response.status_code} - {response.text}"
            raise Exception(error_msg)

    elif response.status_code != 200:
        error_msg = f"Error: {response.status_code} - {response.text}"
        raise Exception(error_msg)

    # Process the response
    data = response.json()

    # Get total number of examples
    total_examples = 0
    if "dataset_info" in data:
        # Handle different response structures (config keys or direct access)
        if "en" in data["dataset_info"]:
            # Handle multi-config format like in your first example
            config_data = data["dataset_info"]["en"]
        elif "default" in data["dataset_info"]:
            # Handle default config format like in your second example
            config_data = data["dataset_info"]["default"]
        else:
            # Handle direct access format like in the original documentation
            config_data = data["dataset_info"]

        # Sum examples across all splits
        if "splits" in config_data:
            for split_name, split_info in config_data["splits"].items():
                if "num_examples" in split_info:
                    total_examples += split_info["num_examples"]

    result = {"dataset_name": dataset_name, "total_examples": total_examples}

    return result


def get_dataset_properties(dataset_name):
    """
    Get column names from any Hugging Face dataset using only the first configuration.

    Args:
        dataset_name (str): Dataset name on Hugging Face

    Returns:
        list: Column names of the dataset
    """
    logger.info(f"Getting properties for dataset: {dataset_name}")

    # Create a temporary directory for caching
    temp_dir = tempfile.mkdtemp()
    original_cache = os.environ.get("HF_DATASETS_CACHE")
    logger.info(f"Created temporary directory for cache: {temp_dir}")

    try:
        # Set cache to temporary directory
        os.environ["HF_DATASETS_CACHE"] = temp_dir

        # Step 1: Identify available configurations
        configs = []
        try:
            logger.info("Attempting to get dataset config names")
            configs = get_dataset_config_names(dataset_name)
            logger.info(f"Found configs: {configs}")
        except Exception as e:
            logger.warning(f"Error getting config names directly: {e}")
            # Try to extract configs from error message
            error_msg = str(e)
            if "Config name is missing" in error_msg:
                logger.info("Extracting configs from error message")
                config_match = re.search(r"available configs: \[(.*?)\]", error_msg)
                if config_match:
                    configs_str = config_match.group(1)
                    configs = [c.strip("'\" ") for c in configs_str.split(",")]
                    logger.info(f"Extracted configs from error: {configs}")

        # Step 2: Use only the first config if multiple configs exist
        config_to_use = None
        if configs:
            config_to_use = configs[0]
            logger.info(f"Using only the first config: {config_to_use}")

        # Step 3: Try multiple methods to get schema information

        # Method 1: Try using dataset builder
        logger.info("Trying dataset builder method...")
        try:
            if config_to_use:
                logger.info(f"Using builder with config: {config_to_use}")
                builder = load_dataset_builder(dataset_name, config_to_use)
            else:
                logger.info("Using builder without config")
                builder = load_dataset_builder(dataset_name)

            if (
                hasattr(builder, "info")
                and hasattr(builder.info, "features")
                and builder.info.features
            ):
                logger.info("Successfully got features from builder")
                return list(builder.info.features.keys())
        except Exception as e:
            logger.warning(f"Builder method failed: {e}")

        # Method 2: Try streaming method
        logger.info("Trying streaming method...")
        columns = get_schema_streaming(dataset_name, config_to_use)
        if columns:
            logger.info("Successfully got columns via streaming")
            return columns

        # Method 3: Last resort - try minimal load
        logger.info("Trying minimal load method...")
        columns = get_schema_minimal_load(dataset_name, config_to_use)
        if columns:
            logger.info("Successfully got columns via minimal load")
            return columns

        # If all methods failed
        logger.error("All methods failed to get dataset properties")
        return []

    finally:
        # Clean up
        if original_cache:
            os.environ["HF_DATASETS_CACHE"] = original_cache
        else:
            os.environ.pop("HF_DATASETS_CACHE", None)

        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp directory: {e}")


def get_schema_streaming(dataset_name, config=None):
    """Helper to get schema using streaming to minimize data download"""
    for split in ["train", "test", "validation"]:
        try:
            logger.info(
                f"Trying streaming for split '{split}'"
                + (f" with config '{config}'" if config else "")
            )

            if config:
                dataset = load_dataset(
                    dataset_name, config, split=split, streaming=True
                )
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=True)

            logger.info("Dataset loaded in streaming mode")

            # Try getting the first example
            try:
                logger.info("Attempting to get first example from stream")
                example = next(iter(dataset))
                columns = list(example.keys())
                logger.info(f"Got {len(columns)} columns: {columns}")
                return columns
            except StopIteration:
                logger.warning(f"Dataset stream was empty for split '{split}'")
            except Exception as e:
                logger.warning(f"Error getting first example: {e}")

        except Exception as e:
            logger.warning(
                f"Error loading dataset in streaming mode for split '{split}': {e}"
            )

    logger.warning("All splits failed in streaming mode")
    return None


def get_schema_minimal_load(dataset_name, config=None):
    """Helper to get schema using minimal data load"""
    for split in ["train", "test", "validation"]:
        try:
            logger.info(
                f"Trying minimal load for split '{split}'"
                + (f" with config '{config}'" if config else "")
            )

            # Try with a specific slice notation to limit download
            slice_spec = f"{split}[:1]"
            if config:
                dataset = load_dataset(dataset_name, config, split=slice_spec)
            else:
                dataset = load_dataset(dataset_name, split=slice_spec)

            logger.info(f"Minimal dataset loaded, length: {len(dataset)}")

            if len(dataset) > 0:
                columns = list(dataset[0].keys())
                logger.info(f"Got {len(columns)} columns from first example: {columns}")
                return columns
            elif hasattr(dataset, "features") and dataset.features:
                columns = list(dataset.features.keys())
                logger.info(f"Got {len(columns)} columns from features: {columns}")
                return columns
            else:
                logger.warning("Dataset loaded but no examples or features found")

        except Exception as e:
            logger.warning(f"Error with minimal load for split '{split}': {e}")

    logger.warning("All splits failed in minimal load mode")
    return None
