import json
import traceback
import sys
import os
import threading
import gradio as gr
from translator.parser.dynamic import DynamicDataParser
from translator.callback.huggingface import HuggingFaceCallback
from translator.callback.gradio import LogCaptureCallback
from engine.ollama import OllamaEngine
from engine.groq import GroqEngine
from config.qa import QAConfig
from config.cot import COTConfig

# Global storage for active translation processes
active_translation = {"parser": None, "cancellation_event": None}


def validate_output_directory(output_dir, log_callback):
    """
    Validate if the output directory is valid and writable.

    Args:
        output_dir (str): Path to the output directory
        log_callback: Log callback for status updates

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Normalize path
        norm_path = os.path.normpath(output_dir)

        # Check if directory exists, if not create it
        if not os.path.exists(norm_path):
            log_callback.add_log(
                f"Output directory doesn't exist. Creating: {norm_path}"
            )
            os.makedirs(norm_path, exist_ok=True)

        # Verify it's a directory
        if not os.path.isdir(norm_path):
            log_callback.add_log(f"Error: {norm_path} is not a directory")
            return False

        # Check if writable by trying to create a temporary file
        test_file = os.path.join(norm_path, "test_write_permission.tmp")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except (IOError, PermissionError):
            log_callback.add_log(
                f"Error: Cannot write to {norm_path}. Check permissions."
            )
            return False

        return True
    except Exception as e:
        log_callback.add_log(f"Error validating output directory: {str(e)}")
        return False


def cancel_translation(log_file_path):
    """
    Cancel the running translation process.

    Args:
        log_file_path: Path to the log file

    Returns:
        Updated component states
    """
    # Initialize log capture
    log_callback = LogCaptureCallback()
    log_callback.set_components(log_file_path)

    log_callback.add_log("❌ Cancellation requested by user")

    # Check if we have an active translation
    if active_translation["cancellation_event"]:
        # Set the cancellation event to signal threads to stop
        active_translation["cancellation_event"].set()
        log_callback.add_log("❌ Cancellation signal sent to all workers")

        # If we have a parser reference, try to cancel it directly
        if active_translation["parser"]:
            try:
                active_translation["parser"].cancel()
                log_callback.add_log("❌ Cancel signal sent to parser")
            except Exception as e:
                log_callback.add_log(f"Error while cancelling parser: {str(e)}")
        log_callback.add_log(
            "⏳ Cancellation in progress - waiting for workers to terminate..."
        )
    else:
        log_callback.add_log("No active translation found to cancel")

    # Clean up active translation references
    active_translation["parser"] = None
    active_translation["cancellation_event"] = None

    # Re-enable all UI components
    # 4 buttons with primary variant
    button_updates = [gr.update(interactive=True, variant="primary") for _ in range(4)]
    # 21 other components just enabled
    other_updates = [gr.update(interactive=True) for _ in range(21)]
    # Update button visibility
    submit_visible = gr.update(visible=True)
    cancel_visible = gr.update(visible=False)

    log_callback.add_log("✅ UI components have been re-enabled")

    # Return updates for all components
    return (
        [log_callback.get_logs()]
        + button_updates
        + other_updates
        + [submit_visible, cancel_visible]
    )


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
    output_dir,
    log_file_path,
    progress_status=None,
):
    """
    Main function to translate the dataset.
    Note: This function doesn't disable components - that's done by a separate function.
    It only handles the translation process and re-enabling components when done.
    """
    # Initialize log capture
    log_callback = LogCaptureCallback()
    log_callback.set_components(log_file_path, progress_status)

    # Create cancellation event
    cancellation_event = threading.Event()

    # Store in global state for access by cancel button
    active_translation["cancellation_event"] = cancellation_event

    try:
        # Clear log file to start fresh
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("")

        # Write initial log
        log_callback.add_log("Starting dataset translation...")
        log_callback.add_log(
            "UI components have been disabled during translation process."
        )

        # Validate output directory
        if not validate_output_directory(output_dir, log_callback):
            log_callback.add_log("Translation aborted due to invalid output directory.")
            # Re-enable all components with appropriate updates
            # 4 buttons with primary variant
            button_updates = [
                gr.update(interactive=True, variant="primary") for _ in range(4)
            ]
            # 21 other components just enabled
            other_updates = [gr.update(interactive=True) for _ in range(21)]
            # Update button visibility
            submit_visible = gr.update(visible=True)
            cancel_visible = gr.update(visible=False)
            return (
                [log_callback.get_logs()]
                + button_updates
                + other_updates
                + [submit_visible, cancel_visible]
            )

        # Parse field mappings
        try:
            field_mappings = json.loads(field_mappings_str)
        except json.JSONDecodeError:
            log_callback.add_log(
                "Error: Invalid field mappings format. Please provide valid JSON."
            )
            # Re-enable all components with appropriate updates
            button_updates = [
                gr.update(interactive=True, variant="primary") for _ in range(4)
            ]
            other_updates = [gr.update(interactive=True) for _ in range(21)]
            # Update button visibility
            submit_visible = gr.update(visible=True)
            cancel_visible = gr.update(visible=False)
            return (
                [log_callback.get_logs()]
                + button_updates
                + other_updates
                + [submit_visible, cancel_visible]
            )

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
            # Re-enable all components
            button_updates = [
                gr.update(interactive=True, variant="primary") for _ in range(4)
            ]
            other_updates = [gr.update(interactive=True) for _ in range(21)]
            # Update button visibility
            submit_visible = gr.update(visible=True)
            cancel_visible = gr.update(visible=False)
            return (
                [log_callback.get_logs()]
                + button_updates
                + other_updates
                + [submit_visible, cancel_visible]
            )

        # Determine data source
        actual_dataset_name = dataset_name if data_source_type == "dataset" else None
        actual_file_path = file_path if data_source_type == "file" else None

        # Normalize output path
        normalized_output_path = os.path.normpath(output_dir)
        log_callback.add_log(f"Output directory: {normalized_output_path}")

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
            # Create the parser instance with cancellation event
            parser = DynamicDataParser(
                file_path=actual_file_path,
                output_path=normalized_output_path,
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
                cancellation_event=cancellation_event,  # Pass cancellation event
            )

            # Store parser reference for cancellation
            active_translation["parser"] = parser

            # Run the parser (check for cancellation between steps)
            parser.read()

            if cancellation_event.is_set():
                log_callback.add_log(
                    "❌ Translation was cancelled during reading phase"
                )
            else:
                parser.convert()

                if cancellation_event.is_set():
                    log_callback.add_log(
                        "❌ Translation was cancelled during conversion phase"
                    )
                else:
                    parser.save()

                    if cancellation_event.is_set():
                        log_callback.add_log(
                            "❌ Translation was cancelled during saving phase"
                        )
                    else:
                        log_callback.add_log("✅ Translation completed successfully!")
        finally:
            # Restore stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            # Clear the active translation references
            active_translation["parser"] = None
            active_translation["cancellation_event"] = None

        # Re-enable all components on success
        button_updates = [
            gr.update(interactive=True, variant="primary") for _ in range(4)
        ]
        other_updates = [gr.update(interactive=True) for _ in range(21)]

        # Update button visibility
        submit_visible = gr.update(visible=True)
        cancel_visible = gr.update(visible=False)

        # Return the logs output and component updates
        return (
            [log_callback.get_logs()]
            + button_updates
            + other_updates
            + [submit_visible, cancel_visible]
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        log_callback.add_log(f"Error: {str(e)}")
        log_callback.add_log(error_trace)

        # Clear the active translation references
        active_translation["parser"] = None
        active_translation["cancellation_event"] = None

        # Re-enable all components on error
        button_updates = [
            gr.update(interactive=True, variant="primary") for _ in range(4)
        ]
        other_updates = [gr.update(interactive=True) for _ in range(21)]

        # Update button visibility
        submit_visible = gr.update(visible=True)
        cancel_visible = gr.update(visible=False)

        # Return the logs output and component updates
        return (
            [log_callback.get_logs()]
            + button_updates
            + other_updates
            + [submit_visible, cancel_visible]
        )
