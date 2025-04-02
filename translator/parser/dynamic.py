from typing import List, Dict, Optional, Union
from datasets import load_dataset

from config.qa import QAConfig
from translator.parser.base import BaseParser
from translator.parser.utils.config import (
    get_mandatory_fields,
    validate_mappings,
    patch_target_config,
    restore_target_config,
)
from translator.parser.utils.mapper import dynamic_mapping
from translator.parser.utils.memory import (
    estimate_item_memory_usage,
    determine_optimal_batch_size,
    sample_dataset,
)
from translator.parser.utils.translation import translate_batch
from translator.parser.utils.batch import BatchProcessor

from translator.preprocessing.utils import get_dataset_info


class DynamicDataParser(BaseParser):
    def __init__(
        self,
        file_path: Optional[str],
        output_path: str,
        dataset_name: str,
        field_mappings: dict,
        target_config=QAConfig,
        additional_fields: Optional[dict] = None,
        translate_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        do_translate: bool = True,
        batch_size: Optional[int] = None,
        auto_batch_size: bool = True,
        max_memory_percent: float = 0.2,
        min_batch_size: int = 10,
        max_batch_size: int = 5000,
        **parser_kwargs,
    ):
        """
        Initialize a configurable parser for dynamic dataset mapping and translation.
        """
        # Identify mandatory fields from the target_config
        self.mandatory_fields = get_mandatory_fields(target_config)

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
        validate_mappings(field_mappings, self.mandatory_fields)

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
        self._original_get_keys, self._original_annotations = patch_target_config(
            target_config, self.additional_config_fields
        )

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
            restore_target_config(
                target_config, self._original_get_keys, self._original_annotations
            )

        self.dataset_name = dataset_name
        # TODO: add support for local files
        if not self.file_path:
            self.total_record = get_dataset_info(dataset_name, api_token=None).get(
                "total_examples", 0
            )
        self.field_mappings = field_mappings
        self.additional_fields = additional_fields
        self.limit = limit
        self.auto_batch_size = auto_batch_size
        self.batch_size = batch_size
        self.max_memory_percent = max_memory_percent
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.processed_count = 0
        self.sample_size = 0
        self.item_memory_estimate = None

    def read(self) -> None:
        """
        Read data from the specified dataset or local JSON files.
        """
        super(DynamicDataParser, self).read()

        # If file_path is provided, read from local JSON files
        if self.file_path:
            try:
                from translator.reader.json import get_json_reader

                # Get dataset from JSON files
                self.data_read = get_json_reader(
                    self.file_path, recursive=True, verbose=self.verbose
                )

            except Exception as e:
                raise RuntimeError(
                    f"Error reading JSON files from '{self.file_path}': {str(e)}"
                )
        else:
            # Use Hugging Face dataset loading when file_path is None
            try:
                self.data_read = load_dataset(self.dataset_name, streaming=True)
            except Exception as e:
                raise RuntimeError(
                    f"Error loading dataset '{self.dataset_name}': {str(e)}"
                )

        return None

    def convert(self) -> None:
        """
        Set up the conversion process but doesn't materialize the entire dataset.
        """
        super(DynamicDataParser, self).convert()

        # Instead of collecting all data, we'll create generators for each split
        self.data_generators = {}
        for split in self.data_read:
            self.data_generators[split] = dynamic_mapping(
                self.data_read[split],
                self.field_mappings,
                self.mandatory_fields,
                self.verbose,
                self.id_generator,
                self.limit,
                self.additional_fields,
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

        # Use the translation utils to perform the actual translation
        translator_instance = self.get_translator if translator is None else translator

        result = translate_batch(
            data_to_translate,
            self.fields_to_translate,
            translator_instance,
            self.source_lang,
            self.target_lang,
            self.max_example_length,
            self.fail_translation_code,
            self.verbose,
            self.enable_sub_task_thread,
            self.max_list_length_per_thread,
            desc,
        )

        if self.parser_callbacks and (en_data is None and large_chunk is None):
            for callback in self.parser_callbacks:
                callback.on_finish_translate(self)

        return result

    def process_and_save(self) -> None:
        if not hasattr(self, "data_generators") or not self.data_generators:
            raise ValueError("Must call convert() before process_and_save()")

        # Create a function to get a fresh generator for the primary split
        def get_fresh_generator():
            return dynamic_mapping(
                self.data_read[self.primary_split],
                self.field_mappings,
                self.mandatory_fields,
                self.verbose,
                self.id_generator,
                self.limit,
                self.additional_fields,
            )

        # Determine batch size if needed - using a separate generator for sampling
        if self.auto_batch_size and self.batch_size is None:
            try:
                # Use a separate generator instance for sampling
                sample_generator = get_fresh_generator()
                samples = sample_dataset(sample_generator, 20)

                if samples:
                    self.item_memory_estimate = estimate_item_memory_usage(samples)
                    self.batch_size = determine_optimal_batch_size(
                        self.item_memory_estimate,
                        self.max_memory_percent,
                        self.min_batch_size,
                        self.max_batch_size,
                    )
                    print(f"Auto-determined batch size: {self.batch_size}")
                else:
                    self.batch_size = 100
                    print(f"Using default batch size: {self.batch_size}")
            except Exception as e:
                print(f"Error determining batch size: {str(e)}")
                self.batch_size = 100
        elif self.batch_size is None:
            self.batch_size = 100

        # Make sure batch_size is not None to avoid the error you encountered
        assert self.batch_size is not None, "Batch size must not be None"

        # Create a fresh generator for actual processing
        processing_generator = get_fresh_generator()

        # Create processor and define the processing function
        batch_processor = BatchProcessor(
            self.output_dir,
            self.parser_name,
            self.batch_size,
            self.item_memory_estimate or 1000,
            self.max_memory_percent,
            self.min_batch_size,
            self.max_batch_size,
            parser_callbacks=self.parser_callbacks,
            total_records=getattr(
                self, "total_record", None
            ),  # Pass the total_record if available
            limit=getattr(self, "limit", None),
        )

        # Process and save the data using the fresh generator
        batch_processor.process_and_save(
            processing_generator,
            lambda batch, desc: self.translate_converted(en_data=batch, desc=desc)
            if self.do_translate
            else batch,
            self.limit,
        )

    def save(self) -> None:
        """
        Process and save the data incrementally.
        """
        self.process_and_save()
