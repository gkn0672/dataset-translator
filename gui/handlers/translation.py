import json
import traceback
import sys
import os
import time
import io
import threading
import gradio as gr
from translator.parser.dynamic import DynamicDataParser
from translator.callback.verbose import VerboseCallback
from translator.callback.huggingface import HuggingFaceCallback
from translator.callback.gradio import LogCaptureCallback
from engine.ollama import OllamaEngine
from engine.groq import GroqEngine
from config.qa import QAConfig
from config.cot import COTConfig


class CaptureOutput:
    """Capture stdout and stderr to log file and memory"""

    def __init__(self, log_file, log_list):
        self.log_file = log_file
        self.log_list = log_list
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.buffer = io.StringIO()
        self.lock = threading.Lock()

    def write(self, text):
        # Write to original stdout for IDE console
        self.stdout.write(text)
        self.stdout.flush()

        if text.strip():  # Ignore empty lines
            # Add timestamp for logs
            timestamp = time.strftime("[%H:%M:%S]")

            with self.lock:
                # Write to log file with timestamp for each line
                with open(self.log_file, "a", encoding="utf-8") as f:
                    for line in text.splitlines():
                        if line.strip():
                            f.write(f"{timestamp} {line.strip()}\n")
                    f.flush()

                # Store in memory for final return
                for line in text.splitlines():
                    if line.strip():
                        self.log_list.append(f"{timestamp} {line.strip()}")

    def flush(self):
        self.stdout.flush()

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr


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
    log_file_path,  # Path to the log file
    progress_status=None,  # Progress status component
):
    """
    Main function to translate the dataset.
    """
    # Clear log file to start fresh
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("")

    # Initialize stored logs list
    stored_logs = []

    # Use a context manager to capture all output
    with CaptureOutput(log_file_path, stored_logs):
        try:
            print("Starting dataset translation...")

            # Parse field mappings
            try:
                field_mappings = json.loads(field_mappings_str)
            except json.JSONDecodeError:
                print(
                    "Error: Invalid field mappings format. Please provide valid JSON."
                )
                return "\n".join(stored_logs)

            # Determine which target config to use
            print(f"Using target config: {target_config}")
            config_class = QAConfig if target_config == "QAConfig" else COTConfig

            # Set up callbacks
            parser_callbacks = []
            if push_to_huggingface:
                parser_callbacks.append(HuggingFaceCallback)

            # Create translator engine
            if translator_engine == "ollama":
                translator = OllamaEngine(model_name=translator_model)
                print(f"Using Ollama translator with model: {translator_model}")
            elif translator_engine == "groq":
                # Use GroqEngine
                translator = GroqEngine()
                print("Using Groq translator")
            else:
                print(
                    f"Unknown translator engine: {translator_engine}. Please use Ollama or Groq."
                )
                return "\n".join(stored_logs)

            # Determine data source
            actual_dataset_name = (
                dataset_name if data_source_type == "dataset" else None
            )
            actual_file_path = file_path if data_source_type == "file" else None

            # Log configuration
            print("Configuration:")
            print(f"  Data Source Type: {data_source_type}")
            print(f"  Dataset Name: {actual_dataset_name}")
            print(f"  File Path: {actual_file_path}")
            print(f"  Target Config: {target_config}")
            print(f"  Field Mappings: {field_mappings}")
            print(
                f"  Memory: {max_memory_percent * 100}%, Batch Size: {min_batch_size}-{max_batch_size}"
            )

            # Create the parser
            print("Creating DynamicDataParser...")

            parser = DynamicDataParser(
                file_path=actual_file_path,
                output_path="C:\\Code\\dataset-translator\\samples\\out",
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
            print("Reading dataset...")
            parser.read()

            print("Converting data...")
            parser.convert()

            print("Translating and saving data...")
            parser.save()

            print("Translation completed successfully!")

            # Return the logs
            return "\n".join(stored_logs)

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error: {str(e)}")
            print(error_trace)
            return "\n".join(stored_logs)
