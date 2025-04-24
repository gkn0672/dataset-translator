import gradio as gr
from gui.utils.io import render_additional_fields
from gui.utils.field import update_remove_field_dropdown


def add_field(key, value, additional_fields):
    """
    Add a new field to the additional fields list.

    Args:
        key (str): Field key name
        value (str): Field value
        additional_fields (list): Current list of additional fields

    Returns:
        tuple: Tuple of updated components
    """
    # Check if key is empty
    if not key:
        return (
            key,
            gr.update(),
            additional_fields,
            render_additional_fields(additional_fields),
            update_remove_field_dropdown(additional_fields),
        )

    # Check if key already exists
    if any(field["key"] == key for field in additional_fields):
        return (
            key,
            gr.update(),
            additional_fields,
            render_additional_fields(additional_fields),
            update_remove_field_dropdown(additional_fields),
        )

    # Add the new field
    additional_fields.append({"key": key, "value": value})

    # Return updated values
    return (
        "",  # Clear the key input
        gr.update(),  # Keep the value dropdown as is
        additional_fields,  # Updated fields list
        render_additional_fields(additional_fields),  # Updated HTML table
        update_remove_field_dropdown(additional_fields),  # Updated removal dropdown
    )


def remove_field(field_key, additional_fields):
    """
    Remove a field from the additional fields list.

    Args:
        field_key (str): Key of the field to remove
        additional_fields (list): Current list of additional fields

    Returns:
        tuple: Tuple of updated components
    """
    if not field_key:
        return (
            additional_fields,
            render_additional_fields(additional_fields),
            update_remove_field_dropdown(additional_fields),
        )

    # Remove the field with the matching key
    additional_fields = [
        field for field in additional_fields if field["key"] != field_key
    ]

    return (
        additional_fields,  # Updated fields list
        render_additional_fields(additional_fields),  # Updated HTML table
        update_remove_field_dropdown(additional_fields),  # Updated removal dropdown
    )


def update_dropdown_options(
    target_config,
    qa_question,
    qa_answer,
    cot_question,
    cot_reasoning,
    cot_answer,
    available_properties,
):
    """
    Update dropdown options to exclude already selected properties.

    Args:
        target_config (str): Configuration type
        qa_question (str): QA question field value
        qa_answer (str): QA answer field value
        cot_question (str): COT question field value
        cot_reasoning (str): COT reasoning field value
        cot_answer (str): COT answer field value
        available_properties (list): List of all available properties

    Returns:
        tuple: Updated dropdown options for each field
    """
    # Get all selected values
    selected_values = []

    if target_config == "QAConfig":
        if qa_question:
            selected_values.append(qa_question)
        if qa_answer:
            selected_values.append(qa_answer)
    else:  # COTConfig
        if cot_question:
            selected_values.append(cot_question)
        if cot_reasoning:
            selected_values.append(cot_reasoning)
        if cot_answer:
            selected_values.append(cot_answer)

    # For each dropdown, filter out selected values except its own value
    qa_question_options = [""]
    qa_answer_options = [""]
    cot_question_options = [""]
    cot_reasoning_options = [""]
    cot_answer_options = [""]

    if available_properties:
        qa_question_options = [""] + [
            prop
            for prop in available_properties
            if prop not in selected_values or prop == qa_question
        ]
        qa_answer_options = [""] + [
            prop
            for prop in available_properties
            if prop not in selected_values or prop == qa_answer
        ]
        cot_question_options = [""] + [
            prop
            for prop in available_properties
            if prop not in selected_values or prop == cot_question
        ]
        cot_reasoning_options = [""] + [
            prop
            for prop in available_properties
            if prop not in selected_values or prop == cot_reasoning
        ]
        cot_answer_options = [""] + [
            prop
            for prop in available_properties
            if prop not in selected_values or prop == cot_answer
        ]

    # Return updates for all dropdowns
    return (
        gr.update(choices=qa_question_options, value=qa_question),
        gr.update(choices=qa_answer_options, value=qa_answer),
        gr.update(choices=cot_question_options, value=cot_question),
        gr.update(choices=cot_reasoning_options, value=cot_reasoning),
        gr.update(choices=cot_answer_options, value=cot_answer),
        # Also return available options for the new field value dropdown
        gr.update(
            choices=[""]
            + [prop for prop in available_properties if prop not in selected_values]
            if available_properties
            else [""],
        ),
    )
