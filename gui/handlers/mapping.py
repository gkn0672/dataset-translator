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
