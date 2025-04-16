import gradio as gr
import os
import tempfile
from gradio_log import Log


def create_logs():
    """
    Create the logs output component using gradio_log.
    """
    components = {}

    # Create a temporary log file that will persist during the session
    log_file_path = os.path.join(tempfile.gettempdir(), "dataset_translator_logs.txt")

    # Clear any previous logs
    with open(log_file_path, "w") as f:
        f.write("")

    # Create the Log component with increased height for better display
    components["logs_output"] = Log(
        log_file=log_file_path,
        dark=True,  # Match the dark theme
        xterm_font_size=12,
        elem_id="logs-output",
        height=400,  # Increased height for better visibility
    )

    # Store the log file path for the callback to use
    components["log_file_path"] = gr.State(log_file_path)

    # Store progress component as State to avoid UI issues
    components["progress_status"] = gr.State("")

    return components
