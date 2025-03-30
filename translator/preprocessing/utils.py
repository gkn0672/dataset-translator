import sys
import time
import socket
import requests

sys.path.insert(0, r"./")
from functools import wraps
from tqdm.auto import tqdm


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
