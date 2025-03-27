import os
import json
import threading
import concurrent.futures
import time
import psutil
from tqdm.auto import tqdm
from typing import List, Optional, Callable, Any


class BatchProcessor:
    """
    Handles processing data in batches with memory management and multi-threading.
    """

    def __init__(
        self,
        output_dir: str,
        parser_name: str,
        batch_size: int,
        item_memory_estimate: float,
        max_memory_percent: float = 0.2,
        min_batch_size: int = 10,
        max_batch_size: int = 5000,
        num_workers: Optional[int] = None,
        parser_callbacks: Optional[List[Any]] = None,
    ):
        """
        Initialize batch processor.

        Args:
            output_dir: Directory to save output
            parser_name: Name of parser (used for filename)
            batch_size: Initial batch size
            item_memory_estimate: Estimated memory per item in bytes
            max_memory_percent: Maximum memory to use (as percentage of available)
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            num_workers: Number of worker threads (None for auto)
            parser_callbacks: List of parser callbacks
        """
        self.output_dir = output_dir
        self.parser_name = parser_name
        self.batch_size = batch_size
        self.item_memory_estimate = item_memory_estimate
        self.max_memory_percent = max_memory_percent
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.parser_callbacks = parser_callbacks

        # Initialize output file
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(f"{output_dir}/{parser_name}.jsonl", "w", encoding="utf-8") as f:
            f.write("")  # Create empty file or truncate existing file

        # Determine number of threads
        if num_workers is None:
            self.num_workers = min(os.cpu_count() or 4, 8)
        else:
            self.num_workers = num_workers

        # Thread synchronization
        self.file_lock = threading.Lock()
        self.memory_lock = threading.Lock()

        # Memory tracking
        self.memory_high_water_mark = 0.0
        self.memory_warning_threshold = 0.7
        self.memory_critical_threshold = 0.8
        self.max_memory_bytes = psutil.virtual_memory().total * max_memory_percent
        self.current_batch_size = batch_size
        self.min_batch_size = max(10, batch_size // 10)
        self.estimated_memory_usage = 0

        # Counter for processed items
        self.processed_counter = self._create_counter()

        # Status monitoring
        self.stop_monitor = threading.Event()

    def _create_counter(self):
        """Create a thread-safe counter"""

        class Counter:
            def __init__(self):
                self.count = 0
                self.lock = threading.Lock()

            def increment(self, amount=1):
                with self.lock:
                    self.count += amount
                    return self.count

        return Counter()

    def update_memory_usage(self, change_in_bytes, is_increase=True):
        """Thread-safe memory usage tracker"""
        with self.memory_lock:
            if is_increase:
                self.estimated_memory_usage += change_in_bytes
            else:
                self.estimated_memory_usage -= change_in_bytes

            # Update high water mark if needed
            self.memory_high_water_mark = max(
                self.memory_high_water_mark, self.estimated_memory_usage
            )

            # Calculate current memory usage ratio
            memory_ratio = self.estimated_memory_usage / self.max_memory_bytes

            # Adaptive batch size adjustment based on memory pressure
            if memory_ratio > self.memory_critical_threshold:
                # Severe memory pressure - reduce batch size significantly
                self.current_batch_size = max(
                    self.min_batch_size, self.current_batch_size // 2
                )
                print(
                    f"⚠️ High memory pressure ({memory_ratio:.1%}). Reducing batch size to {self.current_batch_size}"
                )
            elif memory_ratio > self.memory_warning_threshold:
                # Moderate memory pressure - reduce batch size slightly
                self.current_batch_size = max(
                    self.min_batch_size, int(self.current_batch_size * 0.8)
                )
                print(
                    f"⚠️ Elevated memory pressure ({memory_ratio:.1%}). Adjusting batch size to {self.current_batch_size}"
                )
            elif memory_ratio < 0.4 and self.current_batch_size < self.batch_size:
                # Low memory pressure - gradually increase batch size back toward original
                self.current_batch_size = min(
                    self.batch_size, int(self.current_batch_size * 1.2)
                )

            return memory_ratio

    def _memory_monitor(self):
        """Continuously monitor actual system memory usage"""
        while not self.stop_monitor.is_set():
            actual_memory_percent = psutil.virtual_memory().percent / 100

            # If system memory usage is getting too high overall (not just our estimate)
            if actual_memory_percent > 0.9:  # 90% total system memory
                with self.memory_lock:
                    # Emergency batch size reduction
                    self.current_batch_size = max(
                        self.min_batch_size, self.current_batch_size // 2
                    )
                    print(
                        f"⚠️ SYSTEM MEMORY CRITICAL: {actual_memory_percent:.1%}. Emergency batch size reduction to {self.current_batch_size}"
                    )

            time.sleep(1)  # Check every second

    def process_batch(self, batch, batch_num, process_func):
        """
        Process a batch of data in a thread-safe manner with memory tracking

        Args:
            batch: Batch of data to process
            batch_num: Batch number (for logging)
            process_func: Function to process the batch

        Returns:
            Number of items processed
        """
        if not batch:
            return 0

        batch_size_bytes = len(batch) * self.item_memory_estimate
        # Register memory usage for this batch
        self.update_memory_usage(batch_size_bytes, is_increase=True)

        processed_items = 0

        try:
            # Process the batch with the provided function
            processed_batch = process_func(batch, f"Batch {batch_num}")

            # Thread-safe file writing
            with self.file_lock:
                with open(
                    f"{self.output_dir}/{self.parser_name}.jsonl",
                    "a",
                    encoding="utf-8",
                ) as f:
                    for item in processed_batch:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        processed_items += 1

            # Update the global counter
            self.processed_counter.increment(processed_items)

            return processed_items
        finally:
            # Release memory regardless of success or failure
            self.update_memory_usage(batch_size_bytes, is_increase=False)

    def process_and_save(
        self, data_generator, process_func: Callable, limit: Optional[int] = None
    ):
        """
        Process and save a dataset in batches

        Args:
            data_generator: Generator yielding data items
            process_func: Function to process each batch (e.g., translation)
            limit: Optional limit on number of items to process
        """
        # Create a progress bar
        total_limit = limit if limit else "unknown"
        progress_bar = tqdm(desc=f"Processing (limit: {total_limit})", unit="items")

        # Start memory monitor in background
        monitor_thread = threading.Thread(target=self._memory_monitor, daemon=True)
        monitor_thread.start()

        # Collect and process data in batches using ThreadPoolExecutor
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                futures = []
                batch = []
                batch_num = 0
                items_read = 0

                # Adaptive backpressure - how many batches can be pending
                max_pending_batches = self.num_workers * 3  # Start with default

                # Main processing loop - collect items into batches and submit to thread pool
                for item in data_generator:
                    batch.append(item)
                    items_read += 1

                    # When batch is full, check memory and submit to thread pool
                    if len(batch) >= self.current_batch_size:
                        # Check if we need to wait due to memory pressure
                        memory_ratio = self.update_memory_usage(
                            0, is_increase=False
                        )  # Just check current ratio

                        # Adaptive backpressure based on memory pressure
                        if memory_ratio > self.memory_warning_threshold:
                            max_pending_batches = max(
                                self.num_workers, int(self.num_workers * 1.5)
                            )
                        elif memory_ratio > 0.6:
                            max_pending_batches = self.num_workers * 2
                        else:
                            max_pending_batches = self.num_workers * 3

                        # Wait if we have too many pending batches
                        while len(futures) >= max_pending_batches:
                            # Check completed futures and update progress
                            completed_futures = self._check_completed_futures(
                                futures, progress_bar
                            )

                            # Remove completed futures
                            for future in completed_futures:
                                futures.remove(future)

                            # If still too many pending, wait a bit
                            if len(futures) >= max_pending_batches:
                                time.sleep(0.1)

                        # Submit batch to executor
                        future = executor.submit(
                            self.process_batch, batch, batch_num, process_func
                        )
                        futures.append(future)
                        batch = []  # Start a new batch
                        batch_num += 1

                    # Check if we've reached the limit
                    if limit and items_read >= limit:
                        break

                    # Periodically check and update progress
                    if (
                        len(futures) > 0
                        and items_read % (self.current_batch_size * 2) == 0
                    ):
                        # Check completed futures and update progress
                        completed_futures = self._check_completed_futures(
                            futures, progress_bar
                        )

                        # Remove completed futures
                        for future in completed_futures:
                            futures.remove(future)

                # Process any remaining items in the last batch
                if batch:
                    future = executor.submit(
                        self.process_batch, batch, batch_num, process_func
                    )
                    futures.append(future)

                # Wait for all remaining futures to complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        items_processed = future.result()
                        progress_bar.update(items_processed)
                    except Exception as e:
                        print(f"Error processing batch: {str(e)}")
        finally:
            # Stop memory monitor
            self.stop_monitor.set()
            monitor_thread.join(timeout=1.0)  # Wait for monitor to finish

        # Close progress bar and print summary
        progress_bar.close()
        print(f"Processed and saved {self.processed_counter.count} items")
        print(
            f"Peak memory usage estimate: {self.memory_high_water_mark / (1024**2):.2f} MB"
        )
        print(f"Memory limit: {self.max_memory_bytes / (1024**2):.2f} MB")

        # Call any completion callbacks
        if self.parser_callbacks:
            for callback in self.parser_callbacks:
                if hasattr(callback, "on_finish_save"):
                    callback.on_finish_save(self)

    def _check_completed_futures(self, futures, progress_bar):
        """Check completed futures and update progress"""
        completed_futures = []
        for future in futures:
            if future.done():
                try:
                    items_processed = future.result()
                    progress_bar.update(items_processed)
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                completed_futures.append(future)

        return completed_futures
