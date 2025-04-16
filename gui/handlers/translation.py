import json
import traceback
import sys
from translator.parser.dynamic import DynamicDataParser
from translator.callback.huggingface import HuggingFaceCallback
from translator.callback.gradio import LogCaptureCallback
from engine.ollama import OllamaEngine
from engine.groq import GroqEngine
from config.qa import QAConfig
from config.cot import COTConfig


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
    # Initialize log capture
    log_callback = LogCaptureCallback()
    log_callback.set_components(log_file_path, progress_status)

    try:
        # Clear log file to start fresh
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("")

        # Write initial log
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

        # Set up callbacks - use only LogCaptureCallback for cleaner logs
        parser_callbacks = []  # <-- Changed from [LogCaptureCallback]
        if push_to_huggingface:
            parser_callbacks.append(HuggingFaceCallback)

        # Create translator engine
        if translator_engine == "ollama":
            translator = OllamaEngine(model_name=translator_model)
            log_callback.add_log(
                f"Using Ollama translator with model: {translator_model}"
            )
        elif translator_engine == "groq":
            # Use GroqEngine
            translator = GroqEngine()
            log_callback.add_log("Using Groq translator")
        else:
            log_callback.add_log(
                f"Unknown translator engine: {translator_engine}. Please use Ollama or Groq."
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

        # Setup log capture for direct logging from the process
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Create a custom stdout/stderr redirector that logs to our callback
        class LogRedirector:
            def __init__(self, callback):
                self.callback = callback

            def write(self, text):
                if text.strip():
                    for line in text.splitlines():
                        if line.strip():
                            # Print to original stdout for IDE console
                            original_stdout.write(line + "\n")
                            self.callback.add_log(line.strip())

            def flush(self):
                original_stdout.flush()

        # Redirect stdout and stderr to our log capture
        sys.stdout = LogRedirector(log_callback)
        sys.stderr = LogRedirector(log_callback)

        try:
            parser = DynamicDataParser(
                file_path=actual_file_path,
                output_path="C:\\Code\\dataset-translator\\samples\\out",
                dataset_name=actual_dataset_name,
                field_mappings=field_mappings,
                target_config=config_class,
                do_translate=True,
                translator=translator,
                verbose=use_verbose,
                parser_callbacks=parser_callbacks,  # No LogCaptureCallback here
                limit=int(limit) if limit else None,
                max_memory_percent=float(max_memory_percent),
                min_batch_size=int(min_batch_size),
                max_batch_size=int(max_batch_size),
            )

            # Run the parser
            parser.read()
            parser.convert()
            parser.save()

            log_callback.add_log("Translation completed successfully!")
        finally:
            # Restore stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        # Return the logs
        return log_callback.get_logs()

    except Exception as e:
        error_trace = traceback.format_exc()
        log_callback.add_log(f"Error: {str(e)}")
        log_callback.add_log(error_trace)
        return log_callback.get_logs()
