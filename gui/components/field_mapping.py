import gradio as gr
import json


def create_field_mapping():
    """
    Create the field mapping components.

    Returns:
        dict: Dictionary of field mapping components
    """
    components = {}

    # State for additional fields
    components["additional_fields_state"] = gr.State([])

    with gr.Row(equal_height=False):
        # Field mapping panel - larger portion
        with gr.Column(scale=7):
            # Main field mappings section
            with gr.Group():
                gr.HTML("🔄 Field Mappings")
                gr.HTML("Map dataset properties to configuration fields")

                # QAConfig fields
                components["qa_fields"] = gr.Group(visible=True)
                with components["qa_fields"]:
                    with gr.Row():
                        with gr.Column(scale=2, min_width=100):
                            gr.HTML("Question")
                        with gr.Column(scale=5):
                            components["qa_question_dropdown"] = gr.Dropdown(
                                choices=[""],
                                value="",
                                label=None,
                                allow_custom_value=True,
                                container=True,
                            )
                    with gr.Row():
                        with gr.Column(scale=2, min_width=100):
                            gr.HTML("Answer")
                        with gr.Column(scale=5):
                            components["qa_answer_dropdown"] = gr.Dropdown(
                                choices=[""],
                                value="",
                                label=None,
                                allow_custom_value=True,
                                container=True,
                            )

                # COTConfig fields
                components["cot_fields"] = gr.Group(visible=False)
                with components["cot_fields"]:
                    with gr.Row():
                        with gr.Column(scale=2, min_width=100):
                            gr.HTML("Question")
                        with gr.Column(scale=5):
                            components["cot_question_dropdown"] = gr.Dropdown(
                                choices=[""],
                                value="",
                                label=None,
                                allow_custom_value=True,
                            )
                    with gr.Row():
                        with gr.Column(scale=2, min_width=100):
                            gr.HTML("Reasoning")
                        with gr.Column(scale=5):
                            components["cot_reasoning_dropdown"] = gr.Dropdown(
                                choices=[""],
                                value="",
                                label=None,
                                allow_custom_value=True,
                            )
                    with gr.Row():
                        with gr.Column(scale=2, min_width=100):
                            gr.HTML("Answer")
                        with gr.Column(scale=5):
                            components["cot_answer_dropdown"] = gr.Dropdown(
                                choices=[""],
                                value="",
                                label=None,
                                allow_custom_value=True,
                            )

            # Additional fields section
            with gr.Group():
                gr.HTML("➕ Additional Fields")

                # Display current additional fields
                components["additional_fields_table"] = gr.HTML(
                    """
                    No additional fields added yet
                    """,
                    elem_id="additional-fields-table",
                )

                # Add field section with better layout
                with gr.Row():
                    with gr.Column(scale=2):
                        components["new_field_key"] = gr.Textbox(
                            label="Field Key",
                            placeholder="Enter field name",
                        )
                    with gr.Column(scale=3):
                        components["new_field_value"] = gr.Dropdown(
                            label="Field Value",
                            choices=[""],
                            value="",
                            allow_custom_value=True,
                        )
                    with gr.Column(scale=2):
                        components["add_field_btn"] = gr.Button(
                            "Add Field", variant="primary", size="sm"
                        )

                # Remove field section
                with gr.Row():
                    with gr.Column(scale=4):
                        components["field_to_remove"] = gr.Dropdown(
                            label="Select Field to Remove",
                            choices=[],
                            value=None,
                        )
                    with gr.Column(scale=2):
                        components["remove_field_btn"] = gr.Button(
                            "Remove Field", variant="stop", size="sm"
                        )

        # JSON representation - smaller portion
        with gr.Column(scale=4):
            with gr.Group():
                gr.HTML("📄 JSON Representation")
                components["field_mappings_str"] = gr.Code(
                    label="Field Mappings (Read-only)",
                    language="json",
                    lines=20,
                    value=json.dumps({"question": "", "answer": ""}, indent=2),
                    interactive=False,
                    elem_id="json-editor",
                )

    return components
