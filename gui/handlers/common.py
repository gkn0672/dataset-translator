from gui.handlers.source import (
    update_on_source_change,
    handle_fetch_properties,
    switch_config,
)
from gui.handlers.mapping import add_field, remove_field, update_dropdown_options
from gui.handlers.translation import translate_dataset, cancel_translation
from gui.utils.json import update_json_mapping
import gradio as gr
import os
import threading

# Global app state dictionary
app_state = {"cancellation_event": None}


def disable_all_components():
    """
    Function to disable all components before translation starts.
    This runs immediately when the user clicks 'Start Translation'.
    """
    print("Disabling all components before translation")

    # Create button updates (secondary variant and disabled)
    button_updates = [
        gr.update(interactive=False, variant="secondary"),  # fetch_properties_btn
        gr.update(interactive=False, variant="secondary"),  # add_field_btn
        gr.update(interactive=False, variant="secondary"),  # remove_field_btn
        gr.update(interactive=False, variant="secondary"),  # submit_button
    ]

    # Create other component updates (just disabled)
    other_updates = [gr.update(interactive=False) for _ in range(21)]

    # Return all updates
    return button_updates + other_updates


def update_button_visibility_on_start():
    """Update button visibility when translation starts"""
    return gr.update(visible=False), gr.update(visible=True)


def update_button_visibility_on_cancel():
    """Update button visibility when translation is cancelled"""
    return gr.update(visible=True), gr.update(visible=False)


def register_all_handlers(components):
    """
    Register all event handlers for the application.

    Args:
        components (dict): Dictionary of UI components
    """

    # Source type change
    components["data_source_type"].change(
        lambda source_type, config_type: update_on_source_change(
            source_type, config_type, components
        ),
        inputs=[components["data_source_type"], components["target_config"]],
        outputs=[
            components["dataset_name"],
            components["file_path"],
            components["available_properties_display"],
            components["field_mappings_str"],
        ],
    )

    # Config type change
    components["target_config"].change(
        lambda config_type: switch_config(config_type, components),
        inputs=[components["target_config"]],
        outputs=[
            components["qa_fields"],
            components["cot_fields"],
            components["field_mappings_str"],
        ],
    )

    # Fetch properties button
    components["fetch_properties_btn"].click(
        lambda data_source_type,
        dataset_name,
        file_path,
        target_config: handle_fetch_properties(
            data_source_type, dataset_name, file_path, target_config, components
        ),
        inputs=[
            components["data_source_type"],
            components["dataset_name"],
            components["file_path"],
            components["target_config"],
        ],
        outputs=[
            components["available_properties_display"],
            components["new_field_value"],
            components["field_mappings_str"],
            components["qa_question_dropdown"],
            components["qa_answer_dropdown"],
            components["cot_question_dropdown"],
            components["cot_reasoning_dropdown"],
            components["cot_answer_dropdown"],
            components["additional_fields_state"],
            components["additional_fields_table"],
            components["field_to_remove"],
            components["available_properties_state"],
        ],
    )

    # Add field button
    components["add_field_btn"].click(
        add_field,
        inputs=[
            components["new_field_key"],
            components["new_field_value"],
            components["additional_fields_state"],
        ],
        outputs=[
            components["new_field_key"],
            components["new_field_value"],
            components["additional_fields_state"],
            components["additional_fields_table"],
            components["field_to_remove"],
        ],
    )

    # Remove field button
    components["remove_field_btn"].click(
        remove_field,
        inputs=[components["field_to_remove"], components["additional_fields_state"]],
        outputs=[
            components["additional_fields_state"],
            components["additional_fields_table"],
            components["field_to_remove"],
        ],
    )

    # Update JSON mapping when any field changes
    for input_field in [
        components["qa_question_dropdown"],
        components["qa_answer_dropdown"],
        components["cot_question_dropdown"],
        components["cot_reasoning_dropdown"],
        components["cot_answer_dropdown"],
    ]:
        input_field.change(
            fn=lambda tc, qa_q, qa_a, cot_q, cot_r, cot_a, af, avail_props: (
                update_json_mapping(tc, qa_q, qa_a, cot_q, cot_r, cot_a, af),
                *update_dropdown_options(
                    tc, qa_q, qa_a, cot_q, cot_r, cot_a, avail_props
                ),
            ),
            inputs=[
                components["target_config"],
                components["qa_question_dropdown"],
                components["qa_answer_dropdown"],
                components["cot_question_dropdown"],
                components["cot_reasoning_dropdown"],
                components["cot_answer_dropdown"],
                components["additional_fields_state"],
                components["available_properties_state"],
            ],
            outputs=[
                components["field_mappings_str"],
                components["qa_question_dropdown"],
                components["qa_answer_dropdown"],
                components["cot_question_dropdown"],
                components["cot_reasoning_dropdown"],
                components["cot_answer_dropdown"],
                components["new_field_value"],
            ],
        )

    # Special handling for additional fields state changes
    components["additional_fields_state"].change(
        update_json_mapping,
        inputs=[
            components["target_config"],
            components["qa_question_dropdown"],
            components["qa_answer_dropdown"],
            components["cot_question_dropdown"],
            components["cot_reasoning_dropdown"],
            components["cot_answer_dropdown"],
            components["additional_fields_state"],
        ],
        outputs=[components["field_mappings_str"]],
    )

    # Define component keys for disabling/enabling
    button_keys = [
        "fetch_properties_btn",
        "add_field_btn",
        "remove_field_btn",
        "submit_button",
    ]
    other_keys = [
        "data_source_type",
        "dataset_name",
        "file_path",
        "target_config",
        "qa_question_dropdown",
        "qa_answer_dropdown",
        "cot_question_dropdown",
        "cot_reasoning_dropdown",
        "cot_answer_dropdown",
        "new_field_key",
        "new_field_value",
        "field_to_remove",
        "translator_engine",
        "translator_model",
        "use_verbose",
        "push_to_huggingface",
        "output_dir",
        "limit",
        "max_memory_percent",
        "min_batch_size",
        "max_batch_size",
    ]

    # Update button visibility (make cancel button visible and hide submit button)
    components["submit_button"].click(
        update_button_visibility_on_start,
        outputs=[components["submit_button"], components["cancel_button"]],
        queue=False,  # Run immediately without queueing
    )

    # First click event: Disable all components immediately
    components["submit_button"].click(
        disable_all_components,
        outputs=[components[key] for key in button_keys + other_keys],
        queue=False,  # Run immediately without queueing
    )

    # Second click event: Run the actual translation process
    components["submit_button"].click(
        translate_dataset,
        inputs=[
            components["data_source_type"],
            components["dataset_name"],
            components["file_path"],
            components["field_mappings_str"],
            components["target_config"],
            components["translator_engine"],
            components["translator_model"],
            components["use_verbose"],
            components["push_to_huggingface"],
            components["limit"],
            components["max_memory_percent"],
            components["min_batch_size"],
            components["max_batch_size"],
            components["output_dir"],
            components["log_file_path"],
            components["progress_status"],
        ],
        outputs=[components["logs_output"]]
        + [components[key] for key in button_keys + other_keys]
        + [components["submit_button"], components["cancel_button"]],
        queue=True,  # Queue this function for background execution
        concurrency_limit=1,  # Only one instance at a time
    )

    # Cancel button handler
    components["cancel_button"].click(
        cancel_translation,
        inputs=[components["log_file_path"]],
        outputs=[components["logs_output"]]
        + [components[key] for key in button_keys + other_keys]
        + [components["submit_button"], components["cancel_button"]],
        queue=False,  # Run immediately without queueing
    )
