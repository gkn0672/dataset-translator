import gradio as gr
import json
import sys
import io
import traceback
from dataclasses import fields

# Import required modules
from config.qa import QAConfig
from config.cot import COTConfig
from translator.parser.dynamic import DynamicDataParser
from translator.callback.verbose import VerboseCallback
from translator.callback.huggingface import HuggingFaceCallback
from engine.ollama import OllamaEngine
from translator.callback.gradio import LogCaptureCallback
from translator.preprocessing.utils import get_dataset_properties


# Class to capture stdout for additional logging
class StdoutCapture:
    def __init__(self):
        self.buffer = io.StringIO()
        self.old_stdout = None

    def start(self):
        self.old_stdout = sys.stdout
        sys.stdout = self.buffer

    def stop(self):
        if self.old_stdout:
            sys.stdout = self.old_stdout

    def get_content(self):
        return self.buffer.getvalue()


# Get fields from a config class, excluding 'qas_id'
def get_config_fields(config_class):
    """
    Get field names from a config class.

    Args:
        config_class: QAConfig or COTConfig class (not instance)

    Returns:
        List of field names, excluding 'qas_id'
    """
    all_fields = fields(config_class)
    return [f.name for f in all_fields if f.name != "qas_id"]


# Generate a template mapping with fields from config class
def generate_config_template(config_type):
    """
    Generate a template JSON mapping with fields from the specified config type.
    """
    if config_type == "QAConfig":
        config_fields = get_config_fields(QAConfig)
    elif config_type == "COTConfig":
        config_fields = get_config_fields(COTConfig)
    else:
        return "{}"

    mapping = {field: "" for field in config_fields}
    return json.dumps(mapping, indent=2)


# Function to fetch dataset properties
def fetch_dataset_properties(data_source_type, dataset_name, file_path, config_type):
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


# Main function to process the dataset translation
def translate_dataset(
    data_source_type,
    dataset_name,
    file_path,
    field_mappings_str,
    target_config,
    translator_engine,
    translator_model,
    use_verbose,
    push_to_huggingface,
    limit,
    max_memory_percent,
    min_batch_size,
    max_batch_size,
    progress=gr.Progress(),
):
    # Initialize log capture
    log_callback = LogCaptureCallback()
    stdout_capture = StdoutCapture()

    stdout_capture.start()

    try:
        # Generate initial log
        log_callback.add_log("Starting dataset translation...")

        # Parse field mappings
        try:
            field_mappings = json.loads(field_mappings_str)
        except json.JSONDecodeError:
            log_callback.add_log(
                "Error: Invalid field mappings format. Please provide valid JSON."
            )
            return log_callback.get_logs()

        # Determine which target config to use
        log_callback.add_log(f"Using target config: {target_config}")
        config_class = QAConfig if target_config == "QAConfig" else COTConfig

        # Set up callbacks
        parser_callbacks = [log_callback]
        if use_verbose:
            parser_callbacks.append(VerboseCallback)
        if push_to_huggingface:
            parser_callbacks.append(HuggingFaceCallback)

        # Create translator engine
        if translator_engine == "ollama":
            translator = OllamaEngine(model_name=translator_model)
            log_callback.add_log(
                f"Using Ollama translator with model: {translator_model}"
            )
        elif translator_engine == "groq":
            # Placeholder for GroqEngine
            log_callback.add_log(
                "GroqEngine not implemented yet. Please use Ollama for now."
            )
            return log_callback.get_logs()

        # Determine data source
        actual_dataset_name = dataset_name if data_source_type == "dataset" else None
        actual_file_path = file_path if data_source_type == "file" else None

        # Log configuration
        log_callback.add_log("Configuration:")
        log_callback.add_log(f"  Data Source Type: {data_source_type}")
        log_callback.add_log(f"  Dataset Name: {actual_dataset_name}")
        log_callback.add_log(f"  File Path: {actual_file_path}")
        log_callback.add_log(f"  Target Config: {target_config}")
        log_callback.add_log(f"  Field Mappings: {field_mappings}")
        log_callback.add_log(
            f"  Memory: {max_memory_percent * 100}%, Batch Size: {min_batch_size}-{max_batch_size}"
        )

        # Create the parser
        log_callback.add_log("Creating DynamicDataParser...")

        parser = DynamicDataParser(
            file_path=actual_file_path,
            output_path="./output",
            dataset_name=actual_dataset_name,
            field_mappings=field_mappings,
            target_config=config_class,
            do_translate=True,
            translator=translator,
            verbose=use_verbose,
            parser_callbacks=parser_callbacks,
            limit=int(limit) if limit else None,
            max_memory_percent=float(max_memory_percent),
            min_batch_size=int(min_batch_size),
            max_batch_size=int(max_batch_size),
        )

        # Run the parser
        progress(0.33, "Reading data...")
        parser.read()

        progress(0.66, "Converting data...")
        parser.convert()

        progress(1.0, "Saving translated data...")
        parser.save()

        log_callback.add_log("Translation completed successfully!")

        # Get stdout content
        stdout_content = stdout_capture.get_content()

        # Return the logs
        return (
            log_callback.get_logs() + "\n\n--- Standard Output ---\n" + stdout_content
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        log_callback.add_log(f"Error: {str(e)}")
        log_callback.add_log(error_trace)

        # Get stdout content
        stdout_content = stdout_capture.get_content()

        return (
            log_callback.get_logs() + "\n\n--- Standard Output ---\n" + stdout_content
        )

    finally:
        # Stop capturing stdout
        stdout_capture.stop()


# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Dataset Translator") as app:
        gr.Markdown("# Dataset Translator")
        gr.Markdown(
            "Configure and run DynamicDataParser to translate datasets from English to Vietnamese"
        )

        # Initialize state variables
        available_properties_state = gr.State([])
        additional_fields_state = gr.State(
            []
        )  # For storing additional fields beyond mandatory

        with gr.Row():
            # Left panel for configuration
            with gr.Column(scale=3):
                # Data source selection
                with gr.Group():
                    gr.Markdown("### Data Source")
                    data_source_type = gr.Radio(
                        label="Source Type",
                        choices=["dataset", "file"],
                        value="dataset",
                    )

                    # Dataset input with fetch properties button
                    dataset_name = gr.Textbox(
                        label="Dataset Name",
                        placeholder="e.g., argilla/magpie-ultra-v0.1",
                        value="argilla/magpie-ultra-v0.1",
                        visible=True,
                    )

                    fetch_properties_btn = gr.Button(
                        "Fetch Properties", variant="primary", size="sm", min_width=400
                    )

                    file_path = gr.Textbox(
                        label="File Path",
                        placeholder="e.g., ./data/my_dataset.csv",
                        visible=False,
                    )

                    # Choose Target Config - Moved here before field mappings
                    gr.Markdown("### Target Config")
                    target_config = gr.Dropdown(
                        label="Config Type",
                        choices=["QAConfig", "COTConfig"],
                        value="QAConfig",
                    )

                    # Display available properties as text
                    gr.Markdown("### Available Dataset Properties")
                    available_properties_display = gr.Textbox(
                        label="Available Properties", interactive=False, lines=3
                    )

                # Field mappings
                with gr.Group():
                    gr.Markdown("### Field Mappings")
                    gr.Markdown("Map dataset properties to config fields")

                    # QAConfig fields
                    qa_fields = gr.Group(visible=True)
                    with qa_fields:
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("Question")
                            with gr.Column(scale=2):
                                qa_question_dropdown = gr.Dropdown(
                                    choices=[""],
                                    value="",
                                    label=None,
                                    allow_custom_value=True,
                                )
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("Answer")
                            with gr.Column(scale=2):
                                qa_answer_dropdown = gr.Dropdown(
                                    choices=[""],
                                    value="",
                                    label=None,
                                    allow_custom_value=True,
                                )

                    # COTConfig fields
                    cot_fields = gr.Group(visible=False)
                    with cot_fields:
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("Question")
                            with gr.Column(scale=2):
                                cot_question_dropdown = gr.Dropdown(
                                    choices=[""],
                                    value="",
                                    label=None,
                                    allow_custom_value=True,
                                )
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("Reasoning")
                            with gr.Column(scale=2):
                                cot_reasoning_dropdown = gr.Dropdown(
                                    choices=[""],
                                    value="",
                                    label=None,
                                    allow_custom_value=True,
                                )
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("Answer")
                            with gr.Column(scale=2):
                                cot_answer_dropdown = gr.Dropdown(
                                    choices=[""],
                                    value="",
                                    label=None,
                                    allow_custom_value=True,
                                )

                    # Additional fields section
                    with gr.Group():
                        gr.Markdown("### Additional Fields")

                        # Display current additional fields in a table
                        additional_fields_table = gr.HTML(
                            """<div class="additional-fields-table">
                            <p><i>No additional fields added yet</i></p>
                            </div>""",
                            elem_id="additional-fields-table",
                        )

                        # Add field section
                        with gr.Row():
                            with gr.Column(scale=1):
                                new_field_key = gr.Textbox(
                                    label="Field Key", placeholder="Enter field name"
                                )
                            with gr.Column(scale=2):
                                new_field_value = gr.Dropdown(
                                    label="Field Value",
                                    choices=[""],
                                    value="",
                                    allow_custom_value=True,
                                )
                            with gr.Column(scale=1):
                                add_field_btn = gr.Button(
                                    "Add Field", variant="primary"
                                )

                        # Remove field section
                        with gr.Row():
                            with gr.Column(scale=2):
                                field_to_remove = gr.Dropdown(
                                    label="Select Field to Remove",
                                    choices=[],
                                    value=None,
                                )
                            with gr.Column(scale=1):
                                remove_field_btn = gr.Button(
                                    "Remove Field", variant="stop"
                                )

                    # JSON representation of the mappings
                    gr.Markdown("### JSON Representation (Read-only)")
                    field_mappings_str = gr.Code(
                        label="Field Mappings (JSON)",
                        language="json",
                        lines=10,
                        value='{\n  "question": "",\n  "answer": ""\n}',
                        interactive=False,
                    )

                # Configuration section
                with gr.Group():
                    gr.Markdown("### Configuration")

                    with gr.Row():
                        translator_engine = gr.Dropdown(
                            label="Translator Engine",
                            choices=["ollama", "groq"],
                            value="ollama",
                        )

                    translator_model = gr.Textbox(
                        label="Translator Model",
                        placeholder="e.g., llama3.1:8b-instruct-q4_0",
                        value="llama3.1:8b-instruct-q4_0",
                    )

                    with gr.Row():
                        use_verbose = gr.Checkbox(
                            label="Use Verbose Logging", value=True
                        )

                        push_to_huggingface = gr.Checkbox(
                            label="Push to HuggingFace", value=False
                        )

                # Performance settings
                with gr.Group():
                    gr.Markdown("### Performance Settings")

                    with gr.Row():
                        limit = gr.Number(
                            label="Limit (records)", value=10, precision=0
                        )

                        max_memory_percent = gr.Slider(
                            label="Max Memory %",
                            minimum=0.1,
                            maximum=0.9,
                            value=0.6,
                            step=0.1,
                        )

                    with gr.Row():
                        min_batch_size = gr.Number(
                            label="Min Batch Size", value=1, precision=0
                        )

                        max_batch_size = gr.Number(
                            label="Max Batch Size", value=5, precision=0
                        )

                submit_button = gr.Button(
                    "Start Translation", variant="primary", size="lg"
                )

            # Right panel for logs
            with gr.Column(scale=2):
                logs_output = gr.Textbox(
                    label="Log Output",
                    lines=30,
                    max_lines=30,
                    interactive=False,
                    autoscroll=True,
                )

        # Function to update the JSON mapping based on all fields
        def update_json_mapping(
            target_config,
            qa_question,
            qa_answer,
            cot_question,
            cot_reasoning,
            cot_answer,
            additional_fields,
        ):
            mapping = {}

            # Add mandatory fields based on config type
            if target_config == "QAConfig":
                mapping["question"] = qa_question
                mapping["answer"] = qa_answer
            else:  # COTConfig
                mapping["question"] = cot_question
                mapping["reasoning"] = cot_reasoning
                mapping["answer"] = cot_answer

            # Add additional fields
            for field in additional_fields:
                if (
                    field["key"] and field["key"] not in mapping
                ):  # Only add if key is not empty and not already in mapping
                    mapping[field["key"]] = field["value"]

            return json.dumps(mapping, indent=2)

        # Function to render additional fields as HTML table
        def render_additional_fields(additional_fields):
            if not additional_fields:
                return """<div class="additional-fields-table">
                    <p><i>No additional fields added yet</i></p>
                    </div>"""

            html = """<div class="additional-fields-table">
                <table style="width:100%; border-collapse: collapse;">
                <tr>
                    <th style="text-align:left; width:30%; padding:8px 0; border-bottom:1px solid #ddd;">Field Key</th>
                    <th style="text-align:left; width:70%; padding:8px 0; border-bottom:1px solid #ddd;">Field Value</th>
                </tr>"""

            for field in additional_fields:
                html += f"""<tr>
                    <td style="padding:8px 0; border-bottom:1px solid #eee;">{field["key"]}</td>
                    <td style="padding:8px 0; border-bottom:1px solid #eee;">{field["value"]}</td>
                </tr>"""

            html += "</table></div>"
            return html

        # Function to update the remove field dropdown with current field keys
        def update_remove_field_dropdown(additional_fields):
            field_keys = [field["key"] for field in additional_fields]
            return gr.update(
                choices=field_keys, value=None if not field_keys else field_keys[0]
            )

        # Function to add a new field to additional fields
        def add_field(key, value, additional_fields):
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
                "",
                gr.update(),
                additional_fields,
                render_additional_fields(additional_fields),
                update_remove_field_dropdown(additional_fields),
            )

        # Function to remove a field from additional fields
        def remove_field(field_key, additional_fields):
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
                additional_fields,
                render_additional_fields(additional_fields),
                update_remove_field_dropdown(additional_fields),
            )

        # Function to switch between QA and COT config displays
        def switch_config(config_type):
            if config_type == "QAConfig":
                return {
                    qa_fields: gr.update(visible=True),
                    cot_fields: gr.update(visible=False),
                }
            else:  # COTConfig
                return {
                    qa_fields: gr.update(visible=False),
                    cot_fields: gr.update(visible=True),
                }

        # Function to handle fetching properties
        def fetch_properties(data_source_type, dataset_name, file_path, config_type):
            props_text, json_str, props_list = fetch_dataset_properties(
                data_source_type, dataset_name, file_path, config_type
            )

            # Set dropdown choices - ensure empty string is first
            dropdown_choices = [""] + props_list

            # Parse initial JSON
            try:
                initial_mapping = json.loads(json_str)
            except:
                initial_mapping = {}

            # Extract default values
            qa_question_val = initial_mapping.get("question", "")
            qa_answer_val = initial_mapping.get("answer", "")
            cot_question_val = initial_mapping.get("question", "")
            cot_reasoning_val = initial_mapping.get("reasoning", "")
            cot_answer_val = initial_mapping.get("answer", "")

            # Extract additional fields (fields not in the config)
            additional_fields = []
            if config_type == "QAConfig":
                config_fields = ["question", "answer"]
            else:  # COTConfig
                config_fields = ["question", "reasoning", "answer"]

            for key, value in initial_mapping.items():
                if key not in config_fields:
                    additional_fields.append({"key": key, "value": value})

            # Update dropdowns with new choices and values
            return (
                props_text,  # Display text of properties
                # Update new field value dropdown choices
                gr.update(choices=dropdown_choices, value=""),
                # Update JSON representation
                json_str,
                # Update QA dropdowns
                gr.update(
                    choices=dropdown_choices,
                    value=qa_question_val
                    if qa_question_val in dropdown_choices
                    else "",
                ),
                gr.update(
                    choices=dropdown_choices,
                    value=qa_answer_val if qa_answer_val in dropdown_choices else "",
                ),
                # Update COT dropdowns
                gr.update(
                    choices=dropdown_choices,
                    value=cot_question_val
                    if cot_question_val in dropdown_choices
                    else "",
                ),
                gr.update(
                    choices=dropdown_choices,
                    value=cot_reasoning_val
                    if cot_reasoning_val in dropdown_choices
                    else "",
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

        # Handle showing/hiding dataset_name and file_path based on data_source_type
        def update_on_source_change(source_type, config_type):
            return {
                dataset_name: gr.update(visible=source_type == "dataset"),
                file_path: gr.update(visible=source_type == "file"),
                available_properties_display: gr.update(value=""),
                field_mappings_str: gr.update(
                    value=generate_config_template(config_type)
                ),
            }

        # Connect interface events

        # Source type change
        data_source_type.change(
            update_on_source_change,
            inputs=[data_source_type, target_config],
            outputs=[
                dataset_name,
                file_path,
                available_properties_display,
                field_mappings_str,
            ],
        )

        # Config type change
        target_config.change(
            switch_config, inputs=[target_config], outputs=[qa_fields, cot_fields]
        )

        # Fetch properties button
        fetch_properties_btn.click(
            fetch_properties,
            inputs=[data_source_type, dataset_name, file_path, target_config],
            outputs=[
                available_properties_display,
                new_field_value,  # Update dropdown choices
                field_mappings_str,
                qa_question_dropdown,
                qa_answer_dropdown,
                cot_question_dropdown,
                cot_reasoning_dropdown,
                cot_answer_dropdown,
                additional_fields_state,
                additional_fields_table,
                field_to_remove,
                available_properties_state,
            ],
        )

        # Add field button
        add_field_btn.click(
            add_field,
            inputs=[new_field_key, new_field_value, additional_fields_state],
            outputs=[
                new_field_key,
                new_field_value,
                additional_fields_state,
                additional_fields_table,
                field_to_remove,
            ],
        )

        # Remove field button
        remove_field_btn.click(
            remove_field,
            inputs=[field_to_remove, additional_fields_state],
            outputs=[additional_fields_state, additional_fields_table, field_to_remove],
        )

        # Update JSON mapping when any field changes
        for input_field in [
            qa_question_dropdown,
            qa_answer_dropdown,
            cot_question_dropdown,
            cot_reasoning_dropdown,
            cot_answer_dropdown,
            additional_fields_state,
        ]:
            input_field.change(
                update_json_mapping,
                inputs=[
                    target_config,
                    qa_question_dropdown,
                    qa_answer_dropdown,
                    cot_question_dropdown,
                    cot_reasoning_dropdown,
                    cot_answer_dropdown,
                    additional_fields_state,
                ],
                outputs=[field_mappings_str],
            )

        # Connect the button to the function
        submit_button.click(
            translate_dataset,
            inputs=[
                data_source_type,
                dataset_name,
                file_path,
                field_mappings_str,
                target_config,
                translator_engine,
                translator_model,
                use_verbose,
                push_to_huggingface,
                limit,
                max_memory_percent,
                min_batch_size,
                max_batch_size,
            ],
            outputs=logs_output,
        )

    return app
