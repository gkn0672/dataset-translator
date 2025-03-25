import sys
import inspect
from typing import List, Dict, Any, Optional, Union
from tqdm.auto import tqdm
from datasets import load_dataset

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
        dataset_items: Any,
        field_mappings: Dict[str, str],
        additional_fields: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dynamically maps dataset items to config fields based on provided mappings.
        """
        data_converted = []
        skipped_items = 0

        for data in tqdm(dataset_items, desc="Converting data"):
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

            data_converted.append(data_dict)

        if skipped_items > 0:
            print(
                f"Skipped {skipped_items} items due to missing or empty mandatory fields"
            )

        return data_converted

    def read(self) -> None:
        """
        Read data from the specified dataset.
        """
        super(DynamicDataParser, self).read()
        try:
            self.data_read = load_dataset(self.dataset_name, streaming=True)
        except Exception as e:
            raise RuntimeError(f"Error loading dataset '{self.dataset_name}': {str(e)}")
        return None

    def convert(self) -> None:
        """
        Convert the dataset to the target format using field mappings.
        """
        super(DynamicDataParser, self).convert()

        all_converted_data = []
        for split in self.data_read:
            converted_split = self.dynamic_mapping(
                self.data_read[split], self.field_mappings, self.additional_fields
            )
            all_converted_data.extend(converted_split)

        if self.limit and self.limit < len(all_converted_data):
            self.converted_data = all_converted_data[: self.limit]
        else:
            self.converted_data = all_converted_data

        if len(self.converted_data) == 0:
            raise ValueError(
                "No valid data items found after filtering for mandatory fields"
            )

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

    # Completely override the translate_converted method to handle all fields properly
    def translate_converted(
        self,
        en_data: List[Dict] = None,
        desc: str = None,
        translator=None,
        large_chunk: List[Dict] = None,
    ) -> Union[None, List[Dict]]:
        """
        Completely overridden version of translate_converted to properly handle additional fields.
        """
        if self.parser_callbacks:
            for callback in self.parser_callbacks:
                callback.on_start_translate(self)

        if en_data is None and large_chunk is None:
            # Process all data
            data_to_translate = self.converted_data
        elif large_chunk is not None:
            data_to_translate = large_chunk
        else:
            data_to_translate = en_data

        if not data_to_translate:
            raise ValueError("No data available for translation")

        result = []
        progress_bar = tqdm(
            data_to_translate, desc=f"Translating {desc if desc else ''}"
        )

        for example in progress_bar:
            translated_example = self._translate_example(example, translator)
            result.append(translated_example)

        # Handle the return value based on how the method was called
        if en_data:
            return result
        elif large_chunk:
            if (
                not hasattr(self, "converted_data_translated")
                or self.converted_data_translated is None
            ):
                self.converted_data_translated = result
            else:
                self.converted_data_translated.extend(result)
        else:
            self.converted_data_translated = result

            if self.parser_callbacks:
                for callback in self.parser_callbacks:
                    callback.on_finish_translate(self)

        return None

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

        Args:
            items: List of strings to translate
            translator: Translator instance to use

        Returns:
            List of translated strings
        """
        # Split list into manageable chunks
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
