import sys
import time
import socket

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
