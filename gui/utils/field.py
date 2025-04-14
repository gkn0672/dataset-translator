import gradio as gr
from translator.preprocessing.utils import get_dataset_properties
from gui.utils.json import generate_config_template, get_config_fields
import json
from config.qa import QAConfig
from config.cot import COTConfig


def update_remove_field_dropdown(additional_fields):
    """
    Update the dropdown for removing fields.

    Args:
        additional_fields (list): List of field dictionaries

    Returns:
        gr.update: Gradio update object for the dropdown
    """
    field_keys = [field["key"] for field in additional_fields]
    return gr.update(
        choices=field_keys, value=None if not field_keys else field_keys[0]
    )


def fetch_dataset_properties(data_source_type, dataset_name, file_path, config_type):
    """
    Fetch dataset properties based on source type.

    Args:
        data_source_type (str): Type of data source ("dataset" or "file")
        dataset_name (str): Name of the dataset
        file_path (str): Path to the file
        config_type (str): Type of configuration

    Returns:
        tuple: (properties_str, json_str, properties_list)
    """
    if data_source_type == "dataset" and not dataset_name:
        return "Please enter a dataset name", generate_config_template(config_type), []
    elif data_source_type == "file" and not file_path:
        return "Please enter a file path", generate_config_template(config_type), []

    try:
        if data_source_type == "dataset":
            properties = get_dataset_properties(dataset_name)
        else:  # file
            # Future implementation for file properties
            return (
                "Getting properties from files is not yet implemented",
                generate_config_template(config_type),
                [],
            )

        if not properties:
            return "No properties found", generate_config_template(config_type), []

        # Format properties as comma-separated text
        properties_str = ", ".join(properties)

        # Get required fields for the config type
        if config_type == "QAConfig":
            config_fields = get_config_fields(QAConfig)
        elif config_type == "COTConfig":
            config_fields = get_config_fields(COTConfig)
        else:
            config_fields = []

        # Initialize with config fields
        mapping = {field: "" for field in config_fields}

        # Try to match properties to fields based on name similarity
        for prop in properties:
            prop_lower = prop.lower()
            for field in config_fields:
                if field in prop_lower:
                    mapping[field] = prop
                    break

        # Return the available properties string, initial mapping, and list of properties
        return properties_str, json.dumps(mapping, indent=2), properties

    except Exception as e:
        print(f"Error fetching properties: {e}")
        return f"Error: {str(e)}", generate_config_template(config_type), []
