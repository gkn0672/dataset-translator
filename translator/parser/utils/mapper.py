from typing import Dict, Any, Optional, List, Generator, Iterator
from tqdm.auto import tqdm


def dynamic_mapping(
    dataset_items: Iterator,
    field_mappings: Dict[str, str],
    mandatory_fields: List[str],
    verbose: bool = False,
    id_generator=None,
    limit: Optional[int] = None,
    additional_fields: Optional[Dict[str, str]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Dynamically maps dataset items to config fields based on provided mappings.
    Returns a generator instead of a list.
    """
    skipped_items = 0
    items_processed = 0
    if verbose:
        print("Starting dynamic mapping of dataset items in verbose mode...")
    for data in tqdm(dataset_items, desc="Converting data"):
        # Check if we've reached the limit
        if limit is not None and items_processed >= limit:
            break

        data_dict = {}
        skip_item = False

        # Add ID field if not specified in mappings
        if "qas_id" not in field_mappings and id_generator:
            data_dict["qas_id"] = id_generator()

        # Map fields according to the mapping dictionary
        for config_field, dataset_field in field_mappings.items():
            if dataset_field in data:
                value = data[dataset_field]
                data_dict[config_field] = value

                # Only check mandatory status for fields in the mandatory fields list
                if config_field in mandatory_fields and (value is None or value == ""):
                    if verbose:
                        print(
                            f"Skipping item: Mandatory field '{config_field}' is empty"
                        )
                    skip_item = True
                    skipped_items += 1
                    break
            else:
                # If field is mandatory but missing in data, skip the item
                if config_field in mandatory_fields:
                    if verbose:
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
        print(f"Skipped {skipped_items} items due to missing or empty mandatory fields")
