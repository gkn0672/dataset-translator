import os
import json
import threading
import time
import psutil
import math
from tqdm.auto import tqdm
from typing import List, Optional, Callable, Any, Iterator
from queue import Queue, Empty
from translator.preprocessing.utils import timeit


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
        total_records: Optional[int] = None,
        limit: Optional[int] = None,
        cancellation_event: Optional[threading.Event] = None,
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
            total_records: Total number of records to process (if known)
            limit: Optional limit on number of items to process
            cancellation_event: Event to signal when processing should be cancelled
        """
        self.output_dir = output_dir
        self.parser_name = parser_name
        self.batch_size = batch_size
        self.item_memory_estimate = item_memory_estimate
        self.max_memory_percent = max_memory_percent
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.parser_callbacks = parser_callbacks
        self.total_records = total_records
        self.limit = limit
        self.cancellation_event = cancellation_event
        # Track worker utilization
        self.worker_stats = {}

        # Initialize output file
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(f"{output_dir}/{parser_name}.jsonl", "w", encoding="utf-8") as f:
            f.write("")  # Create empty file or truncate existing file

        # Determine number of threads
        if num_workers is None:
            self.num_workers = min(os.cpu_count() or 4, 8)
        else:
            self.num_workers = num_workers

        print(f"✅ Available workers: {self.num_workers}")

        # Thread synchronization
        self.file_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.batch_queue_lock = threading.Lock()

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

        # Batch tracking
        self.batch_sizes = []

        # Status monitoring
        self.stop_monitor = threading.Event()
        self.processing_complete = threading.Event()

        # Calculate optimal batch distribution based on worker count
        self._calculate_worker_based_distribution()

        # Batch queue for workers
        self.batch_queue = Queue(
            maxsize=self.num_workers * 2
        )  # Buffer 2 batches per worker
        self.active_workers = 0
        self.workers_lock = threading.Lock()

    def _calculate_worker_based_distribution(self):
        """
        Calculate batch size based on dividing total records by number of workers.
        This ensures each worker gets an equal share of the workload.
        """
        if self.total_records is None:
            # If total records is unknown, use default batch size
            print("⚠️ Total records unknown - using default batch size")
            self.worker_batch_size = self.batch_size
            return

        print(
            f"📊 Dividing {self.total_records} records among {self.num_workers} workers"
        )

        # Calculate ideal items per worker (items_per_worker)
        if self.limit:
            self.total_records = min(self.limit, self.total_records)
        items_per_worker = math.ceil(self.total_records / self.num_workers)

        # Check if items_per_worker exceeds max_batch_size
        if items_per_worker > self.max_batch_size:
            print(
                f"⚠️ Items per worker ({items_per_worker}) exceeds max batch size ({self.max_batch_size})"
            )
            print(f"✅ Using max batch size of {self.max_batch_size}")
            self.worker_batch_size = self.max_batch_size
        else:
            print(f"✅ Each worker will process {items_per_worker} items")
            self.worker_batch_size = items_per_worker

        # Calculate expected number of batches
        total_batches = math.ceil(self.total_records / self.worker_batch_size)
        batches_per_worker = total_batches / self.num_workers

        print(f"📦 Will create ~{total_batches} batches total")
        print(f"📦 Each worker will process ~{batches_per_worker:.2f} batches")

        # Use this as our target batch size
        self.current_batch_size = self.worker_batch_size

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

            return memory_ratio

    def _memory_monitor(self):
        """Continuously monitor actual system memory usage"""
        while not self.stop_monitor.is_set():
            # Check if cancellation is requested
            if self.cancellation_event and self.cancellation_event.is_set():
                self.stop_monitor.set()
                break

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

    def track_worker_usage(self, worker_id, batch_size):
        """Track worker utilization statistics"""
        with self.stats_lock:
            if worker_id not in self.worker_stats:
                self.worker_stats[worker_id] = {
                    "batches_processed": 0,
                    "items_processed": 0,
                }

            self.worker_stats[worker_id]["batches_processed"] += 1
            self.worker_stats[worker_id]["items_processed"] += batch_size

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

        # Check if cancellation is requested
        if self.cancellation_event and self.cancellation_event.is_set():
            return 0

        # Get current thread ID for tracking
        worker_id = threading.get_ident()

        batch_size_bytes = len(batch) * self.item_memory_estimate
        # Register memory usage for this batch
        self.update_memory_usage(batch_size_bytes, is_increase=True)

        processed_items = 0

        try:
            # Process the batch with the provided function
            processed_batch = process_func(batch, f"Batch {batch_num}")

            # Check again for cancellation after processing
            if self.cancellation_event and self.cancellation_event.is_set():
                return 0

            # Thread-safe file writing
            with self.file_lock:
                with open(
                    f"{self.output_dir}/{self.parser_name}_vi.json",
                    "a",
                    encoding="utf-8",
                ) as f:
                    for item in processed_batch:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        processed_items += 1

            # Update statistics
            self.track_worker_usage(worker_id, len(batch))
            self.processed_counter.increment(processed_items)

            return processed_items
        finally:
            # Release memory regardless of success or failure
            self.update_memory_usage(batch_size_bytes, is_increase=False)

    def worker_thread(self, process_func, progress_bar):
        """Worker thread that processes batches from the queue"""
        with self.workers_lock:
            self.active_workers += 1

        try:
            while not self.stop_monitor.is_set():
                # Check for cancellation
                if self.cancellation_event and self.cancellation_event.is_set():
                    break

                try:
                    # Get batch from queue, with a timeout to regularly check if we should stop
                    batch_info = self.batch_queue.get(timeout=0.5)
                    if batch_info is None:  # Sentinel value indicating no more work
                        self.batch_queue.task_done()
                        break

                    batch, batch_num = batch_info

                    # Check for cancellation before processing
                    if self.cancellation_event and self.cancellation_event.is_set():
                        self.batch_queue.task_done()
                        break

                    # Process the batch
                    items_processed = self.process_batch(batch, batch_num, process_func)

                    # Update progress if not cancelled
                    if not (
                        self.cancellation_event and self.cancellation_event.is_set()
                    ):
                        progress_bar.update(items_processed)

                    # Mark task as done
                    self.batch_queue.task_done()

                except Empty:
                    # Queue is empty, check if producer is finished
                    if self.processing_complete.is_set():
                        break
                    # Otherwise, just continue waiting
                    continue
        finally:
            with self.workers_lock:
                self.active_workers -= 1

    def cancel(self):
        """Signal all workers to stop processing and clean up resources"""
        print("❌ Cancellation requested - stopping all processing")

        # Set events to stop processing
        self.stop_monitor.set()
        self.processing_complete.set()

        # Clear queue and add sentinel values to ensure workers exit
        try:
            while not self.batch_queue.empty():
                try:
                    self.batch_queue.get_nowait()
                    self.batch_queue.task_done()
                except Empty:
                    break
        except Exception as e:
            print(f"Error clearing queue: {e}")

        # Add sentinel values to ensure all workers exit
        for _ in range(self.num_workers):
            try:
                self.batch_queue.put(None)
            except Exception as e:
                print(f"Error adding sentinel: {e}")

        print("❌ Translation cancelled by user")
        return True

    @timeit
    def process_and_save(
        self,
        data_generator: Iterator,
        process_func: Callable,
        limit: Optional[int] = None,
    ):
        """
        Process and save a dataset in batches

        Args:
            data_generator: Generator yielding data items
            process_func: Function to process each batch (e.g., translation)
            limit: Optional limit on number of items to process
        """
        # Create a progress bar
        total_limit = limit if limit else (self.total_records or "unknown")
        progress_bar = tqdm(desc=f"Processing (limit: {total_limit})", unit="items")

        # Start memory monitor in background
        monitor_thread = threading.Thread(target=self._memory_monitor, daemon=True)
        monitor_thread.start()

        # Start worker threads
        workers = []
        for _ in range(self.num_workers):
            worker = threading.Thread(
                target=self.worker_thread,
                args=(process_func, progress_bar),
                daemon=True,
            )
            worker.start()
            workers.append(worker)

        # Process data in batches, streaming directly to worker queue
        try:
            batch_num = 0
            items_collected = 0
            current_batch = []
            current_batch_size = self.current_batch_size

            # Fill batch queue with work
            for item in data_generator:
                # Check for cancellation
                if self.cancellation_event and self.cancellation_event.is_set():
                    print("❌ Cancellation detected during batch creation")
                    break

                current_batch.append(item)
                items_collected += 1

                # If we've reached the batch size or the limit, process the batch
                if len(current_batch) >= current_batch_size:
                    # Check for cancellation before adding to queue
                    if self.cancellation_event and self.cancellation_event.is_set():
                        break

                    # Put batch in queue for processing (blocks if queue is full)
                    self.batch_queue.put((current_batch, batch_num))

                    self.batch_sizes.append(len(current_batch))
                    batch_num += 1

                    # Start a new batch
                    current_batch = []

                    # Adjust batch size if necessary (based on memory pressure)
                    current_batch_size = self.current_batch_size

                # Check if we've reached the limit
                if limit and items_collected >= limit:
                    break

            # Check for cancellation before processing remaining items
            if not (self.cancellation_event and self.cancellation_event.is_set()):
                # Process any remaining items
                if current_batch:
                    self.batch_queue.put((current_batch, batch_num))
                    self.batch_sizes.append(len(current_batch))

            # Signal that we're done producing batches
            self.processing_complete.set()

            # If cancelled, call cancel method to clean up
            if self.cancellation_event and self.cancellation_event.is_set():
                self.cancel()
            else:
                # Add sentinel values to queue to signal workers to stop
                for _ in range(self.num_workers):
                    self.batch_queue.put(None)

            # Wait for all queued work to be processed
            if not (self.cancellation_event and self.cancellation_event.is_set()):
                self.batch_queue.join()

            # Wait for all workers to finish
            for worker in workers:
                worker.join(timeout=1.0)

        finally:
            # Stop memory monitor
            self.stop_monitor.set()
            monitor_thread.join(timeout=1.0)  # Wait for monitor to finish

        # Close progress bar and print summary
        progress_bar.close()

        # If cancelled, print cancellation message and return
        if self.cancellation_event and self.cancellation_event.is_set():
            print("❌ Processing was cancelled - cleaned up resources")
            return

        # Print batch distribution statistics
        if self.batch_sizes:
            print("\n📊 Batch size statistics:")
            print(f"  - Number of batches: {len(self.batch_sizes)}")
            print(
                f"  - Average batch size: {sum(self.batch_sizes) / len(self.batch_sizes):.1f}"
            )
            print(f"  - Min batch size: {min(self.batch_sizes)}")
            print(f"  - Max batch size: {max(self.batch_sizes)}")

        # Print worker utilization statistics
        if self.worker_stats:
            print("\n🧵 Worker utilization statistics:")
            active_workers = len(self.worker_stats)
            print(f"  - Active workers: {active_workers}/{self.num_workers}")

            for worker_id, stats in self.worker_stats.items():
                print(
                    f"  - Worker {worker_id % 1000}: {stats['batches_processed']} batches, {stats['items_processed']} items"
                )

        print(f"\nProcessed and saved {self.processed_counter.count} items")
        print(
            f"Peak memory usage estimate: {self.memory_high_water_mark / (1024**2):.2f} MB"
        )
        print(f"Memory limit: {self.max_memory_bytes / (1024**2):.2f} MB")
