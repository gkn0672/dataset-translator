import random
import time
import hashlib
from functools import wraps
from threading import Lock
from typing import (
    Any,
    Dict,
    Callable,
    List,
    Union,
    Set,
    Tuple,
)
from collections import deque

from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field, create_model


def throttle(
    calls_per_minute: int,
    break_interval: float = 0,
    break_duration: float = 0,
    jitter: float = 0,
    verbose: bool = False,
    additional_wait_func: Callable[[], int] = None,
) -> Callable:
    """
    Decorator that limits the number of function calls per minute, adds periodic breaks,
    includes jitter, and allows for additional dynamic wait time.

    Args:
        calls_per_minute (int): The maximum number of function calls allowed per minute.
        break_interval (float, optional): The time period (in seconds) after which a break is taken. Defaults to 0 (no break).
        break_duration (float, optional): The duration (in seconds) of the break after the break_interval. Defaults to 0 (no break).
        jitter (float, optional): Maximum amount of random jitter to add to wait times, in seconds. Defaults to 0.
        verbose (bool, optional): If True, prints additional information about the throttling process. Defaults to False.
        additional_wait_func (Callable[[], int], optional): A function that returns an integer number of seconds to be added to the wait time. Defaults to None.

    Returns:
        Callable: The decorated function.

    Example:
        def dynamic_wait():
            return random.randint(1, 5)  # Returns a random wait time between 1 and 5 seconds

        @throttle(calls_per_minute=10, break_interval=120, break_duration=10, jitter=1, verbose=True, additional_wait_func=dynamic_wait)
        def my_function():
            print("Executing my_function")

        my_function()  # Calls to my_function will be throttled with jitter, include a periodic break, and have additional dynamic wait time.
    """
    interval = 60.0 / calls_per_minute
    lock = Lock()
    last_call = [0.0]
    execution_start_time = [0.0]  # Track the start time of execution

    def add_jitter(delay: float) -> float:
        return delay + random.uniform(0, jitter)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal execution_start_time
            with lock:
                current_time = time.time()
                elapsed = current_time - last_call[0]
                base_wait_time = max(interval - elapsed, 0)
                jittered_wait_time = add_jitter(base_wait_time)

                # Add additional wait time if the function is provided
                if additional_wait_func:
                    additional_wait = additional_wait_func()
                    jittered_wait_time += additional_wait
                    if verbose:
                        print(
                            f"Adding {additional_wait} seconds of additional wait time"
                        )

                # Initialize execution_start_time if it's the first call
                if execution_start_time[0] == 0.0:
                    execution_start_time[0] = current_time

                # Check for periodic break
                if break_interval > 0:
                    time_since_start = current_time - execution_start_time[0]
                    if time_since_start >= break_interval:
                        jittered_break_duration = add_jitter(break_duration)
                        if verbose:
                            print(
                                f"Taking a break for {jittered_break_duration:.4f} seconds after {break_interval} seconds of execution."
                            )
                        time.sleep(jittered_break_duration)
                        execution_start_time[0] = (
                            time.time()
                        )  # Reset execution start time after break

                if jittered_wait_time > 0:
                    if verbose:
                        print(
                            f"Throttling: waiting for {jittered_wait_time:.4f} seconds before calling {func.__name__}"
                        )
                    time.sleep(jittered_wait_time)

                last_call[0] = time.time()
                if verbose:
                    print(
                        f"Calling function {func.__name__} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_call[0]))}"
                    )
                return func(*args, **kwargs)

        return wrapper

    return decorator


def brust_throttle(
    calls_per_minute: int, verbose: bool = False, extra_delay: float = 1.25
) -> Callable:
    """
    Decorator function that limits the number of calls to a function per minute.
    Args:
        calls_per_minute (int): The maximum number of calls allowed per minute.
        verbose (bool, optional): If True, print messages when rate limit is exceeded. Defaults to False.
        extra_delay (float, optional): Additional delay in seconds to wait after rate limit is exceeded. Defaults to 1.25.
    Returns:
        Callable: The decorated function.
    Example:
        @brust_throttle(calls_per_minute=10, verbose=True, extra_delay=0.5)
        def my_function():
            # function implementation
    """
    last_calls = deque()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()

            # Clean up old calls that are outside the current time window
            while last_calls and current_time - last_calls[0] > 60:
                last_calls.popleft()

            # Wait if the call limit has been reached
            if len(last_calls) >= calls_per_minute:
                wait_time = 60 - (current_time - last_calls[0]) + extra_delay
                if verbose:
                    print(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds.")
                time.sleep(wait_time)
                current_time = time.time()

                # Clean up old calls again after waiting
                while last_calls and current_time - last_calls[0] > 60:
                    last_calls.popleft()

            # Record the current call
            last_calls.append(current_time)
            if verbose:
                print(
                    f"Calling function {func.__name__} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def hash_input(value: Union[str, List[str]], hash: bool = True) -> str:
    """
    Hashes the input value using MD5 algorithm and returns the hash value as a string.
    Parameters:
        value (Union[str, List[str]]): The value to be hashed. It can be either a string or a list of strings.
        hash (bool, optional): Specifies whether to hash the input value or not. Defaults to True.
    Returns:
        str: The hash value as a string.
    Raises:
        ValueError: If the input value is a list and contains elements that are not strings.
    """

    if isinstance(value, list):
        # Ensure all elements in the list are strings
        if not all(isinstance(item, str) for item in value):
            raise ValueError("All elements of the list must be strings.")
        value = "".join(value) + f"list_{len(value)}"
    elif not isinstance(value, str):
        value = str(value)

    return hashlib.md5(value.encode("utf-8")).hexdigest() if hash else value


def pop_half_set(s: Set) -> Tuple[Set, Set]:
    """Pop half of the elements from the set s."""
    num_to_pop = len(s) // 2
    popped_elements: Set = set()

    for _ in range(num_to_pop):
        popped_elements.add(s.pop())

    return popped_elements, s


def pop_half_dict(d: Dict) -> Tuple[Dict, Dict]:
    """Pop half of the elements from the dictionary d."""
    num_to_pop = len(d) // 2
    keys_to_pop = list(d.keys())[:num_to_pop]
    popped_elements: Dict = {}

    for key in keys_to_pop:
        popped_elements[key] = d.pop(key)

    return popped_elements, d


def create_dynamic_model(model_name: str, fields: Dict[str, Any]) -> BaseModel:
    """
    Create a dynamic model with the given model name and fields.

    Args:
        model_name (str): The name of the model.
        fields (Dict[str, Any]): A dictionary containing the fields of the model.

    Returns:
        BaseModel: The dynamically created model.
    """
    return create_model(model_name, **fields)


def fuzzy_match(
    input_string, comparison_strings: list, threshold=80, disable_fuzzy: bool = False
):
    """
    Check if the input string matches any of the comparison strings using fuzzy matching.
    Parameters:
    - input_string (str): The input string to be matched.
    - comparison_strings (list): A list of strings to compare against the input string.
    - threshold (int, optional): The minimum fuzzy match ratio required for a match. Defaults to 80.
    - disable_fuzzy (bool, optional): If True, disable fuzzy matching and only perform exact string comparison. Defaults to False.
    Returns:
    - bool: True if a match is found, False otherwise.
    """

    for comparison_string in comparison_strings:
        if (
            fuzz.ratio(input_string, comparison_string) >= threshold
            and not disable_fuzzy
        ):
            return True
        else:
            if input_string == comparison_string:
                return True
    return False


if __name__ == "__main__":
    fields = {
        "name": (
            str,
            Field(..., description="The name of the person"),
        ),  # Required field with description
        "age": (
            int,
            Field(None, description="The age of the person"),
        ),  # Optional field with description
        "email": (
            str,
            Field(None, description="The email address of the person"),
        ),  # Optional field with description
    }

    DynamicModel = create_dynamic_model("DynamicModel", fields)

    # Create an instance of the dynamic model
    instance = DynamicModel(name="John Doe", age=30)
    print(instance)

    print(DynamicModel.model_json_schema())
