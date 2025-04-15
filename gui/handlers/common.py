from gui.handlers.source import (
    update_on_source_change,
    handle_fetch_properties,
    switch_config,
)
from gui.handlers.mapping import add_field, remove_field, update_dropdown_options
from gui.handlers.translation import translate_dataset
from gui.utils.json import update_json_mapping


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

    # Connect the submit button
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
        ],
        outputs=components["logs_output"],
    )
