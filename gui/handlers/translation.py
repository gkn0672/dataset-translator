import json
import traceback
import gradio as gr
from gui.utils.io import StdoutCapture
from translator.parser.dynamic import DynamicDataParser
from translator.callback.verbose import VerboseCallback
from translator.callback.huggingface import HuggingFaceCallback
from translator.callback.gradio import LogCaptureCallback
from engine.ollama import OllamaEngine
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
    progress=gr.Progress(),
):
    """
    Main function to translate the dataset.

    Args:
        Various parameters for dataset translation configuration

    Returns:
        str: Log output
    """
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
        parser_callbacks = [LogCaptureCallback]
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
