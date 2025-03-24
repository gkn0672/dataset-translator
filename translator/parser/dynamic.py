import sys
import inspect
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
from datasets import load_dataset

sys.path.insert(0, r"./")
from config.qa import QAConfig
from translator.parser.base import BaseParser
from translator.callback import VerboseCallback
from engine.ollama import OllamaEngine


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

        Args:
            file_path: Input file path (can be a dummy file for dataset loading)
            output_path: Directory to save output files
            dataset_name: HuggingFace dataset name to load
            field_mappings: Dict mapping config field names to dataset field names
            target_config: Configuration class (dataclass) defining required fields
            additional_fields: Optional dict of additional fields not in config to include
            translate_fields: Optional list of fields to translate (defaults to all mappable fields)
            limit: Optional limit on number of examples to process
            do_translate: Whether to perform translation
            **parser_kwargs: Additional arguments to pass to DataParser
        """
        # Identify mandatory fields from the target_config
        self.mandatory_fields = self._get_mandatory_fields(target_config)

        # Ensure all mandatory fields are in the field_mappings
        self._validate_mappings(field_mappings, self.mandatory_fields)

        # If translate_fields not specified, use all fields from mappings that are in the config
        if translate_fields is None:
            translate_fields = [
                field
                for field in field_mappings.keys()
                if field in target_config.get_keys() and field != "qas_id"
            ]

        parser_name = (
            dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
        )

        super().__init__(
            file_path,
            output_path,
            parser_name=parser_name,
            target_config=target_config,
            target_fields=translate_fields,
            do_translate=do_translate,
            **parser_kwargs,
        )

        self.dataset_name = dataset_name
        self.field_mappings = field_mappings
        self.additional_fields = additional_fields
        self.limit = limit

    def _get_mandatory_fields(self, config_class) -> List[str]:
        """
        Identify mandatory fields from a dataclass-based configuration.

        Args:
            config_class: The configuration class to analyze

        Returns:
            List of mandatory field names
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

        Args:
            field_mappings: Dict mapping config field names to dataset field names
            mandatory_fields: List of mandatory field names

        Raises:
            ValueError: If a mandatory field is missing from the mappings
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
        Ensures mandatory fields are not empty.

        Args:
            dataset_items: Iterable of dataset items to convert
            field_mappings: Dict mapping config field names to dataset field names
            additional_fields: Optional dict of additional fields not in config to include

        Returns:
            List of converted dictionaries with proper field mapping
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

                    # Check if mandatory field is empty or None
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
        # Check if read from local file (file_path is not empty and dataset name is empty)
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


# Example usage
if __name__ == "__main__":
    magpie_parser = DynamicDataParser(
        file_path="./samples/dummy.txt",
        output_path="./samples/out",
        dataset_name="argilla/magpie-ultra-v0.1",
        field_mappings={
            "qas_id": "id",  # Optional: map from dataset if available
            "question": "instruction",
            "answer": "response",
        },
        target_config=QAConfig,
        do_translate=True,
        translator=OllamaEngine(model_name="llama3.1:8b-instruct-q4_0"),
        verbose=True,
        parser_callbacks=[VerboseCallback],
        max_example_per_thread=25,
        large_chunks_threshold=3000,
        limit=100,
    )

    magpie_parser.read()
    magpie_parser.convert()
    magpie_parser.save
