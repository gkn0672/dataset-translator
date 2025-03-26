import sys
import inspect
from typing import List, Dict, Any, Optional, Union, Generator, Iterator
from tqdm.auto import tqdm
from datasets import load_dataset
import psutil

sys.path.insert(0, r"./")
from config.qa import QAConfig
from translator.parser.base import BaseParser


class DynamicDataParser(BaseParser):
    def __init__(
        self,
        file_path: str,
        output_path: str,
        dataset_name: str,
        field_mappings: dict,
        target_config=QAConfig,
        additional_fields: Optional[dict] = None,
        translate_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        do_translate: bool = True,
        batch_size: Optional[
            int
        ] = None,  # Now optional, can be determined automatically
        auto_batch_size: bool = True,  # Whether to dynamically determine batch size
        max_memory_percent: float = 0.2,  # Target memory usage (20% of available RAM)
        min_batch_size: int = 10,  # Minimum batch size regardless of memory
        max_batch_size: int = 5000,  # Maximum batch size regardless of memory
        **parser_kwargs,
    ):
        """
        Initialize a configurable parser for dynamic dataset mapping and translation.
        """
        # Identify mandatory fields from the target_config
        self.mandatory_fields = self._get_mandatory_fields(target_config)

        # Store config fields and mapping info before validation
        self.config_fields = target_config.get_keys()

        # Store all fields that we want to include
        self.all_fields = list(field_mappings.keys())

        # Identify additional fields (not in config)
        self.additional_config_fields = [
            field
            for field in field_mappings.keys()
            if field not in self.config_fields and field != "qas_id"
        ]

        # Ensure all mandatory fields are in the field_mappings
        self._validate_mappings(field_mappings, self.mandatory_fields)

        # If translate_fields not specified, use all fields from mappings except qas_id
        if translate_fields is None:
            translate_fields = [
                field for field in field_mappings.keys() if field != "qas_id"
            ]

        # Store the fields to translate for later use
        self.fields_to_translate = translate_fields

        parser_name = (
            dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
        )

        # The key fix: temporarily modify the target_config to support additional fields
        self._patch_target_config(target_config)

        try:
            super().__init__(
                file_path,
                output_path,
                parser_name=parser_name,
                target_config=target_config,
                target_fields=translate_fields,
                do_translate=do_translate,
                **parser_kwargs,
            )
        finally:
            # Restore the original target_config
            self._restore_target_config(target_config)

        self.dataset_name = dataset_name
        self.field_mappings = field_mappings
        self.additional_fields = additional_fields
        self.limit = limit
        self.auto_batch_size = auto_batch_size
        self.batch_size = batch_size
        self.max_memory_percent = max_memory_percent
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.processed_count = 0
        self.current_batch = []
        self.sample_size = 0
        self.item_memory_estimate = None

    def _patch_target_config(self, target_config):
        """
        Temporarily patch the target_config class to include additional fields.
        This handles both the get_keys method and type annotations.
        """
        # Save original methods and annotations
        self._original_get_keys = target_config.get_keys
        self._original_annotations = getattr(
            target_config, "__annotations__", {}
        ).copy()

        # Create extended get_keys method
        extended_keys = self._original_get_keys() + self.additional_config_fields

        def extended_get_keys(cls):
            return extended_keys

        # Patch the get_keys method
        target_config.get_keys = classmethod(extended_get_keys)

        # Patch the annotations
        # Add string type annotations for additional fields
        updated_annotations = self._original_annotations.copy()
        for field in self.additional_config_fields:
            updated_annotations[field] = str

        # Update the class annotations
        target_config.__annotations__ = updated_annotations

    def _restore_target_config(self, target_config):
        """
        Restore the original target_config class attributes.
        """
        # Restore original get_keys method
        target_config.get_keys = self._original_get_keys

        # Restore original annotations
        target_config.__annotations__ = self._original_annotations

    def _get_mandatory_fields(self, config_class) -> List[str]:
        """
        Identify mandatory fields from a dataclass-based configuration.
        """
        # Get field definitions from the dataclass
        mandatory_fields = []
        for name, param in inspect.signature(config_class.__init__).parameters.items():
            # Skip self parameter
            if name == "self":
                continue

            # If the parameter has no default value (no default=) and no default_factory
            # then it's a mandatory field
            if param.default is param.empty:
                mandatory_fields.append(name)

        return mandatory_fields

    def _validate_mappings(
        self, field_mappings: Dict[str, str], mandatory_fields: List[str]
    ) -> None:
        """
        Validate that all mandatory fields have mappings.
        """
        missing_fields = []
        for field in mandatory_fields:
            # Skip qas_id as it's handled specially if not mapped
            if field != "qas_id" and field not in field_mappings:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"The following mandatory fields are missing from field_mappings: {', '.join(missing_fields)}. "
                f"All mandatory fields ({', '.join(mandatory_fields)}) must be mapped."
            )

    def dynamic_mapping(
        self,
        dataset_items: Iterator,
        field_mappings: Dict[str, str],
        additional_fields: Optional[Dict[str, str]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Dynamically maps dataset items to config fields based on provided mappings.
        Now returns a generator instead of a list.
        """
        skipped_items = 0
        items_processed = 0

        for data in tqdm(dataset_items, desc="Converting data"):
            # Check if we've reached the limit
            if self.limit and items_processed >= self.limit:
                break

            data_dict = {}
            skip_item = False

            # Add ID field if not specified in mappings
            if "qas_id" not in field_mappings:
                data_dict["qas_id"] = self.id_generator()

            # Map fields according to the mapping dictionary
            for config_field, dataset_field in field_mappings.items():
                if dataset_field in data:
                    value = data[dataset_field]
                    data_dict[config_field] = value

                    # Only check mandatory status for fields in the mandatory fields list
                    if config_field in self.mandatory_fields and (
                        value is None or value == ""
                    ):
                        if self.verbose:
                            print(
                                f"Skipping item: Mandatory field '{config_field}' is empty"
                            )
                        skip_item = True
                        skipped_items += 1
                        break
                else:
                    # If field is mandatory but missing in data, skip the item
                    if config_field in self.mandatory_fields:
                        if self.verbose:
                            print(
                                f"Skipping item: Mandatory field '{config_field}' missing in dataset"
                            )
                        skip_item = True
                        skipped_items += 1
                        break
                    # Otherwise set to None
                    data_dict[config_field] = None

            if skip_item:
                continue

            # Add any additional fields that aren't part of the config
            if additional_fields:
                for extra_field, dataset_field in additional_fields.items():
                    if dataset_field in data:
                        data_dict[extra_field] = data[dataset_field]

            items_processed += 1
            yield data_dict

        if skipped_items > 0:
            print(
                f"Skipped {skipped_items} items due to missing or empty mandatory fields"
            )

    def read(self) -> None:
        """
        Read data from the specified dataset.
        """
        # TODO: Add support for reading from files
        super(DynamicDataParser, self).read()
        try:
            self.data_read = load_dataset(self.dataset_name, streaming=True)
        except Exception as e:
            raise RuntimeError(f"Error loading dataset '{self.dataset_name}': {str(e)}")
        return None

    def convert(self) -> None:
        """
        Set up the conversion process but doesn't materialize the entire dataset.
        """
        super(DynamicDataParser, self).convert()

        # Instead of collecting all data, we'll create generators for each split
        self.data_generators = {}
        for split in self.data_read:
            self.data_generators[split] = self.dynamic_mapping(
                self.data_read[split], self.field_mappings, self.additional_fields
            )

        # We'll use the first split as the primary data source
        if self.data_generators:
            self.primary_split = next(iter(self.data_generators.keys()))
            self.converted_data = None  # We no longer store all data in memory
        else:
            raise ValueError("No valid splits found in the dataset")

        return None

    def validate(self, keys: List[str]) -> bool:
        """
        Override BaseParser's validate method to include additional fields.
        """
        # For our dynamic parser, we only need to validate mandatory fields
        for key in self.mandatory_fields:
            if key not in keys:
                raise AssertionError(
                    f"\n Invalid parser, the mandatory key '{key}' is missing from the data.\n"
                    f"you can adjust the fields {self.target_config.__name__} in the 'configs/*.py'"
                    f"  or fill in the missing field."
                )
        return True

    def process_batch(self, batch: List[Dict]) -> None:
        """
        Process a batch of data (translate and save).
        """
        if not batch:
            return

        # Translate the batch if needed
        if self.do_translate:
            translated_batch = self.translate_converted(en_data=batch, desc="Batch")
            self._save_batch(translated_batch)
        else:
            self._save_batch(batch)

    def _save_batch(self, batch: List[Dict]) -> None:
        """
        Save a batch of data to the output file.
        """
        # Implement the save logic here
        # For example:
        import json
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)

        # Append to the output file
        with open(
            f"{self.output_dir}/{self.parser_name}.jsonl", "a", encoding="utf-8"
        ) as f:
            for item in batch:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _estimate_item_memory_usage(self, sample_items: List[Dict]) -> float:
        """
        Estimate memory usage per item based on a sample.

        Args:
            sample_items: A list of sample items to estimate memory usage

        Returns:
            Estimated memory usage per item in bytes
        """
        import sys
        import json

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

    def _determine_optimal_batch_size(self) -> int:
        """
        Determine the optimal batch size based on available memory and sample data.

        Returns:
            Optimal batch size
        """

        # Get available system memory
        available_memory = psutil.virtual_memory().available
        target_memory = available_memory * self.max_memory_percent

        if self.item_memory_estimate is None or self.item_memory_estimate <= 0:
            # If we don't have an estimate, use a default batch size
            return 100

        # Calculate batch size based on memory target
        optimal_batch_size = int(target_memory / self.item_memory_estimate)

        # Apply min/max constraints
        optimal_batch_size = max(
            self.min_batch_size, min(self.max_batch_size, optimal_batch_size)
        )

        return optimal_batch_size

    def _sample_dataset(self, n_samples: int = 20) -> List[Dict]:
        """
        Sample the dataset to estimate memory usage.

        Args:
            n_samples: Number of samples to collect

        Returns:
            List of sample items
        """
        samples = []

        # Get a sample from the primary split
        sample_gen = self.dynamic_mapping(
            self.data_read[self.primary_split],
            self.field_mappings,
            self.additional_fields,
        )

        # Collect samples
        for _ in range(n_samples):
            try:
                item = next(sample_gen)
                samples.append(item)
            except StopIteration:
                break

        return samples

    def process_and_save(self) -> None:
        """
        Process the dataset in batches and save incrementally with multi-threading.
        Uses a simpler threading model with automatic division of work and robust memory management.
        """
        if not hasattr(self, "data_generators") or not self.data_generators:
            raise ValueError("Must call convert() before process_and_save()")

        import os
        import json
        import threading
        import concurrent.futures
        import time
        import psutil
        from tqdm.auto import tqdm

        # File access lock for thread-safe writing
        file_lock = threading.Lock()
        # Memory monitoring lock
        memory_lock = threading.Lock()

        # Initialize the output file
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)
        with open(
            f"{self.output_dir}/{self.parser_name}.jsonl", "w", encoding="utf-8"
        ) as f:
            f.write("")  # Create empty file or truncate existing file

        # Determine batch size if needed
        if self.auto_batch_size and self.batch_size is None:
            try:
                samples = self._sample_dataset(20)
                if samples:
                    self.item_memory_estimate = self._estimate_item_memory_usage(
                        samples
                    )
                    self.batch_size = self._determine_optimal_batch_size()
                    print(f"Auto-determined batch size: {self.batch_size}")
                else:
                    self.batch_size = 100
                    print(f"Using default batch size: {self.batch_size}")
            except Exception as e:
                print(f"Error determining batch size: {str(e)}")
                self.batch_size = 100
        elif self.batch_size is None:
            self.batch_size = 100

        # Determine number of threads based on CPU cores
        num_workers = min(os.cpu_count() or 4, 8)  # Use up to 8 threads or CPU count
        print(f"Processing with {num_workers} threads, batch size {self.batch_size}")

        # Create a thread-safe counter
        class Counter:
            def __init__(self):
                self.count = 0
                self.lock = threading.Lock()

            def increment(self, amount=1):
                with self.lock:
                    self.count += amount
                    return self.count

        processed_counter = Counter()

        # Memory tracking variables
        memory_high_water_mark = 0.0
        memory_warning_threshold = 0.7  # 80% of max_memory_percent
        memory_critical_threshold = 0.8  # 95% of max_memory_percent
        max_memory_bytes = psutil.virtual_memory().total * self.max_memory_percent

        # Adaptive batch size control
        current_batch_size = self.batch_size
        min_batch_size = max(10, self.batch_size // 10)  # Ensure reasonable minimum

        # Current memory usage estimation
        estimated_memory_usage = 0

        def update_memory_usage(change_in_bytes, is_increase=True):
            """Thread-safe memory usage tracker"""
            nonlocal estimated_memory_usage, memory_high_water_mark, current_batch_size

            with memory_lock:
                if is_increase:
                    estimated_memory_usage += change_in_bytes
                else:
                    estimated_memory_usage -= change_in_bytes

                # Update high water mark if needed
                memory_high_water_mark = max(
                    memory_high_water_mark, estimated_memory_usage
                )

                # Calculate current memory usage ratio
                memory_ratio = estimated_memory_usage / max_memory_bytes

                # Adaptive batch size adjustment based on memory pressure
                if memory_ratio > memory_critical_threshold:
                    # Severe memory pressure - reduce batch size significantly
                    current_batch_size = max(min_batch_size, current_batch_size // 2)
                    print(
                        f"⚠️ High memory pressure ({memory_ratio:.1%}). Reducing batch size to {current_batch_size}"
                    )
                elif memory_ratio > memory_warning_threshold:
                    # Moderate memory pressure - reduce batch size slightly
                    current_batch_size = max(
                        min_batch_size, int(current_batch_size * 0.8)
                    )
                    print(
                        f"⚠️ Elevated memory pressure ({memory_ratio:.1%}). Adjusting batch size to {current_batch_size}"
                    )
                elif memory_ratio < 0.4 and current_batch_size < self.batch_size:
                    # Low memory pressure - gradually increase batch size back toward original
                    current_batch_size = min(
                        self.batch_size, int(current_batch_size * 1.2)
                    )

                return memory_ratio

        # Define a thread-safe batch processor that tracks memory
        def process_batch_thread_safe(batch, batch_num):
            """Process a batch of data in a thread-safe manner with memory tracking"""
            if not batch:
                return 0

            batch_size_bytes = len(batch) * self.item_memory_estimate
            # Register memory usage for this batch
            update_memory_usage(batch_size_bytes, is_increase=True)

            processed_items = 0

            try:
                # Translate the batch if needed
                if self.do_translate:
                    translated_batch = self.translate_converted(
                        en_data=batch, desc=f"Batch {batch_num}"
                    )

                    # Thread-safe file writing
                    with file_lock:
                        with open(
                            f"{self.output_dir}/{self.parser_name}.jsonl",
                            "a",
                            encoding="utf-8",
                        ) as f:
                            for item in translated_batch:
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                                processed_items += 1
                else:
                    # For non-translation mode, just save
                    with file_lock:
                        with open(
                            f"{self.output_dir}/{self.parser_name}.jsonl",
                            "a",
                            encoding="utf-8",
                        ) as f:
                            for item in batch:
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                                processed_items += 1

                # Update the global counter
                processed_counter.increment(processed_items)

                return processed_items
            finally:
                # Release memory regardless of success or failure
                update_memory_usage(batch_size_bytes, is_increase=False)

        # Create a progress bar
        total_limit = self.limit if self.limit else "unknown"
        progress_bar = tqdm(desc=f"Processing (limit: {total_limit})", unit="items")

        # Memory monitor thread
        stop_monitor = threading.Event()

        def memory_monitor():
            """Continuously monitor actual system memory usage"""
            while not stop_monitor.is_set():
                actual_memory_percent = psutil.virtual_memory().percent / 100
                target_percent = self.max_memory_percent

                # If system memory usage is getting too high overall (not just our estimate)
                if actual_memory_percent > 0.9:  # 90% total system memory
                    with memory_lock:
                        nonlocal current_batch_size
                        # Emergency batch size reduction
                        current_batch_size = max(
                            min_batch_size, current_batch_size // 2
                        )
                        print(
                            f"⚠️ SYSTEM MEMORY CRITICAL: {actual_memory_percent:.1%}. Emergency batch size reduction to {current_batch_size}"
                        )

                time.sleep(1)  # Check every second

        # Start memory monitor in background
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()

        # Collect and process data in batches using ThreadPoolExecutor
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                futures = []
                batch = []
                batch_num = 0
                items_read = 0

                # Adaptive backpressure - how many batches can be pending
                max_pending_batches = num_workers * 3  # Start with default

                # Main processing loop - collect items into batches and submit to thread pool
                for item in self.data_generators[self.primary_split]:
                    batch.append(item)
                    items_read += 1

                    # When batch is full, check memory and submit to thread pool
                    if len(batch) >= current_batch_size:
                        # Check if we need to wait due to memory pressure
                        memory_ratio = update_memory_usage(
                            0, is_increase=False
                        )  # Just check current ratio

                        # Adaptive backpressure based on memory pressure
                        if memory_ratio > memory_warning_threshold:
                            max_pending_batches = max(
                                num_workers, int(num_workers * 1.5)
                            )
                        elif memory_ratio > 0.6:
                            max_pending_batches = num_workers * 2
                        else:
                            max_pending_batches = num_workers * 3

                        # Wait if we have too many pending batches
                        while len(futures) >= max_pending_batches:
                            # Check completed futures and update progress
                            completed_futures = []
                            for future in futures:
                                if future.done():
                                    try:
                                        items_processed = future.result()
                                        progress_bar.update(items_processed)
                                    except Exception as e:
                                        print(f"Error processing batch: {str(e)}")
                                    completed_futures.append(future)

                            # Remove completed futures
                            for future in completed_futures:
                                futures.remove(future)

                            # If still too many pending, wait a bit
                            if len(futures) >= max_pending_batches:
                                time.sleep(0.1)

                        # Submit batch to executor
                        future = executor.submit(
                            process_batch_thread_safe, batch, batch_num
                        )
                        futures.append(future)
                        batch = []  # Start a new batch
                        batch_num += 1

                    # Check if we've reached the limit
                    if self.limit and items_read >= self.limit:
                        break

                    # Periodically check and update progress (less frequently than in original)
                    if len(futures) > 0 and items_read % (current_batch_size * 2) == 0:
                        # Check completed futures and update progress
                        completed_futures = []
                        for future in futures:
                            if future.done():
                                try:
                                    items_processed = future.result()
                                    progress_bar.update(items_processed)
                                except Exception as e:
                                    print(f"Error processing batch: {str(e)}")
                                completed_futures.append(future)

                        # Remove completed futures
                        for future in completed_futures:
                            futures.remove(future)

                # Process any remaining items in the last batch
                if batch:
                    future = executor.submit(
                        process_batch_thread_safe, batch, batch_num
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
            stop_monitor.set()
            monitor_thread.join(timeout=1.0)  # Wait for monitor to finish

        # Close progress bar and print summary
        progress_bar.close()
        print(f"Processed and saved {processed_counter.count} items")
        print(
            f"Peak memory usage estimate: {memory_high_water_mark / (1024**2):.2f} MB"
        )
        print(f"Memory limit: {max_memory_bytes / (1024**2):.2f} MB")

        # Call any completion callbacks
        if self.parser_callbacks:
            for callback in self.parser_callbacks:
                if hasattr(callback, "on_finish_save"):
                    callback.on_finish_save(self)

    def translate_converted(
        self,
        en_data: List[Dict] = None,
        desc: str = None,
        translator=None,
        large_chunk: List[Dict] = None,
    ) -> Union[None, List[Dict]]:
        """
        Translate a batch of data.
        """
        if self.parser_callbacks:
            for callback in self.parser_callbacks:
                callback.on_start_translate(self)

        if en_data is None and large_chunk is None:
            raise ValueError("Either en_data or large_chunk must be provided")
        elif large_chunk is not None:
            data_to_translate = large_chunk
        else:
            data_to_translate = en_data

        if not data_to_translate:
            return []

        result = []
        progress_bar = tqdm(
            data_to_translate, desc=f"Translating {desc if desc else ''}"
        )

        for example in progress_bar:
            translated_example = self._translate_example(example, translator)
            result.append(translated_example)

        if self.parser_callbacks and (en_data is None and large_chunk is None):
            for callback in self.parser_callbacks:
                callback.on_finish_translate(self)

        return result

    def _translate_example(self, example: Dict, translator=None) -> Dict:
        """
        Translate a single example.

        Args:
            example: The example to translate
            translator: Optional translator instance

        Returns:
            Translated example
        """
        result = example.copy()

        # Get or create a translator instance
        translator_instance = self.get_translator if translator is None else translator

        # Translate each field that should be translated
        for field in self.fields_to_translate:
            if field not in example or example[field] is None or example[field] == "":
                continue

            # Translate the field
            value = example[field]
            if isinstance(value, list):
                # Handle list values
                for i, item in enumerate(value):
                    if len(item) > self.max_example_length:
                        if self.verbose:
                            print(
                                f"Truncating a list item in field {field} as it exceeds max length"
                            )
                        value[i] = item[: self.max_example_length]

                # Check if we need special handling for large lists
                if (
                    self.enable_sub_task_thread
                    and len(value) >= self.max_list_length_per_thread
                ):
                    result[field] = self._translate_large_list(
                        value, translator_instance
                    )
                else:
                    result[field] = translator_instance.translate(
                        value,
                        src=self.source_lang,
                        dest=self.target_lang,
                        fail_translation_code=self.fail_translation_code,
                    )
            else:
                # Handle string values
                if len(value) > self.max_example_length:
                    if self.verbose:
                        print(f"Truncating field {field} as it exceeds max length")
                    value = value[: self.max_example_length]

                result[field] = translator_instance.translate(
                    value,
                    src=self.source_lang,
                    dest=self.target_lang,
                    fail_translation_code=self.fail_translation_code,
                )

        return result

    def _translate_large_list(self, items: List[str], translator) -> List[str]:
        """
        Handle translation of a large list by breaking it into smaller chunks.
        """
        # Implementation remains the same
        chunks = self.split_list(items, self.max_list_length_per_thread)
        results = []

        for chunk in chunks:
            translated = translator.translate(
                chunk,
                src=self.source_lang,
                dest=self.target_lang,
                fail_translation_code=self.fail_translation_code,
            )
            results.extend(translated)

        return results

    # New method to replace the original save method
    def save(self) -> None:
        """
        Process and save the data incrementally.
        """
        self.process_and_save()
