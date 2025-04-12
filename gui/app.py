import gradio as gr
import json
import sys
import io
import traceback

# Import required modules
from config.qa import QAConfig
from config.cot import COTConfig
from translator.parser.dynamic import DynamicDataParser
from translator.callback.verbose import VerboseCallback
from translator.callback.huggingface import HuggingFaceCallback
from engine.ollama import OllamaEngine
from translator.callback.gradio import LogCaptureCallback


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

                    dataset_name = gr.Textbox(
                        label="Dataset Name",
                        placeholder="e.g., argilla/magpie-ultra-v0.1",
                        value="argilla/magpie-ultra-v0.1",
                        visible=True,
                    )

                    file_path = gr.Textbox(
                        label="File Path",
                        placeholder="e.g., ./data/my_dataset.csv",
                        visible=False,
                    )

                # Field mappings
                with gr.Group():
                    gr.Markdown("### Field Mappings")
                    field_mappings_str = gr.Code(
                        label="Field Mappings (JSON)",
                        language="json",
                        lines=7,
                        value='{\n  "question": "instruction",\n  "answer": "response",\n  "intention": "intent"\n}',
                    )

                # Configuration section
                with gr.Group():
                    gr.Markdown("### Configuration")

                    with gr.Row():
                        target_config = gr.Dropdown(
                            label="Target Config",
                            choices=["QAConfig", "COTConfig"],
                            value="QAConfig",
                        )

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

        # Handle showing/hiding dataset_name and file_path based on data_source_type
        def update_visibility(source_type):
            return {
                dataset_name: gr.update(visible=source_type == "dataset"),
                file_path: gr.update(visible=source_type == "file"),
            }

        data_source_type.change(
            update_visibility,
            inputs=[data_source_type],
            outputs=[dataset_name, file_path],
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
