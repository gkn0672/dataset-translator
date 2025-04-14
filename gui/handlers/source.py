import gradio as gr
import json
from gui.utils.field import fetch_dataset_properties
from gui.utils.json import generate_config_template
from gui.utils.io import render_additional_fields
from gui.utils.field import update_remove_field_dropdown


def update_on_source_change(source_type, config_type, components):
    """
    Handle updates when the source type changes.

    Args:
        source_type (str): Type of data source ("dataset" or "file")
        config_type (str): Type of configuration
        components (dict): Dictionary of UI components

    Returns:
        dict: Dictionary of component updates
    """
    return {
        components["dataset_name"]: gr.update(visible=source_type == "dataset"),
        components["file_path"]: gr.update(visible=source_type == "file"),
        components["available_properties_display"]: gr.update(value=""),
        components["field_mappings_str"]: gr.update(
            value=generate_config_template(config_type)
        ),
    }


def handle_fetch_properties(
    data_source_type, dataset_name, file_path, target_config, components
):
    """
    Handle fetching properties from the selected data source.

    Args:
        data_source_type (str): Type of data source ("dataset" or "file")
        dataset_name (str): Name of the dataset
        file_path (str): Path to the file
        target_config (str): Type of configuration
        components (dict): Dictionary of UI components

    Returns:
        tuple: Tuple of component updates
    """
    props_text, json_str, props_list = fetch_dataset_properties(
        data_source_type, dataset_name, file_path, target_config
    )

    # Set dropdown choices - ensure empty string is first
    dropdown_choices = [""] + props_list

    # Parse initial JSON
    try:
        initial_mapping = json.loads(json_str)
    except Exception:
        initial_mapping = {}

    # Extract default values
    qa_question_val = initial_mapping.get("question", "")
    qa_answer_val = initial_mapping.get("answer", "")
    cot_question_val = initial_mapping.get("question", "")
    cot_reasoning_val = initial_mapping.get("reasoning", "")
    cot_answer_val = initial_mapping.get("answer", "")

    # Extract additional fields (fields not in the config)
    additional_fields = []
    if target_config == "QAConfig":
        config_fields = ["question", "answer"]
    else:  # COTConfig
        config_fields = ["question", "reasoning", "answer"]

    for key, value in initial_mapping.items():
        if key not in config_fields:
            additional_fields.append({"key": key, "value": value})

    # Return tuple of updates
    return (
        props_text,  # Display text of properties
        # Update new field value dropdown choices
        gr.update(choices=dropdown_choices, value=""),
        # Update JSON representation
        json_str,
        # Update QA dropdowns
        gr.update(
            choices=dropdown_choices,
            value=qa_question_val if qa_question_val in dropdown_choices else "",
        ),
        gr.update(
            choices=dropdown_choices,
            value=qa_answer_val if qa_answer_val in dropdown_choices else "",
        ),
        # Update COT dropdowns
        gr.update(
            choices=dropdown_choices,
            value=cot_question_val if cot_question_val in dropdown_choices else "",
        ),
        gr.update(
            choices=dropdown_choices,
            value=cot_reasoning_val if cot_reasoning_val in dropdown_choices else "",
        ),
        gr.update(
            choices=dropdown_choices,
            value=cot_answer_val if cot_answer_val in dropdown_choices else "",
        ),
        # Update additional fields
        additional_fields,
        render_additional_fields(additional_fields),
        update_remove_field_dropdown(additional_fields),
        # Update available properties state
        props_list,
    )


def switch_config(config_type, components):
    """
    Handle switching between QA and COT configurations.

    Args:
        config_type (str): Type of configuration
        components (dict): Dictionary of UI components

    Returns:
        dict: Dictionary of component updates
    """
    if config_type == "QAConfig":
        return {
            components["qa_fields"]: gr.update(visible=True),
            components["cot_fields"]: gr.update(visible=False),
        }
    else:  # COTConfig
        return {
            components["qa_fields"]: gr.update(visible=False),
            components["cot_fields"]: gr.update(visible=True),
        }
