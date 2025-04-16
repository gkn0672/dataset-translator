import time
import os
from translator.callback.base import BaseCallback


class LogCaptureCallback(BaseCallback):
    """Callback to capture logs for Gradio UI"""

    def __init__(self):
        self.logs = []
        self.last_update = time.time()
        self.log_file_path = None
        self.progress_component = None

    def set_components(self, log_file_path, progress_component=None):
        """
        Set the log file path and progress component

        Args:
            log_file_path: Path to the log file for gradio_log
            progress_component: The progress status component
        """
        self.log_file_path = log_file_path
        self.progress_component = progress_component

    def add_log(self, message):
        """
        Add a log message with timestamp

        Args:
            message (str): The log message
        """
        # Ensure message has proper newlines
        message = message.strip()
        timestamp = time.strftime("[%H:%M:%S]")
        log_entry = f"{timestamp} {message}"
        self.logs.append(log_entry)

        # Write to log file if available
        if self.log_file_path and os.path.exists(os.path.dirname(self.log_file_path)):
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(f"{log_entry}\n")
                    f.flush()  # Ensure it's written immediately
            except Exception as e:
                print(f"Error writing to log file: {e}")

    def update_progress(self, status_text):
        """Update the progress status text"""
        self.add_log(f"Status: {status_text}")

    def get_logs(self):
        """Get all logs as a string"""
        return "\n".join(self.logs)

    # Each callback method adds proper newlines
    def on_start_init(self, instance):
        self.add_log(f"Parser {instance.parser_name} has started initializing")

    def on_finish_init(self, instance):
        self.add_log(f"Parser {instance.parser_name} has finished initializing")

    def on_start_read(self, instance):
        self.add_log(f"Parser {instance.parser_name} has started reading")

    def on_start_convert(self, instance):
        self.add_log(f"Parser {instance.parser_name} has started converting")

    def on_finish_convert(self, instance):
        self.add_log(f"Parser {instance.parser_name} has finished converting")

    def on_start_save_converted(self, instance):
        self.add_log(
            f"Parser {instance.parser_name} has started saving the converted data"
        )

    def on_finish_save_converted(self, instance):
        self.add_log(
            f"Parser {instance.parser_name} has finished saving the converted data"
        )

    def on_start_translate(self, instance):
        self.add_log(f"Parser {instance.parser_name} has started translating")

    def on_finish_translate(self, instance):
        self.add_log(f"Parser {instance.parser_name} has finished translating")

    def on_error_translate(self, instance, error):
        self.add_log(
            f"Parser {instance.parser_name} has encountered an error during translation: {error}"
        )

    def on_start_save_translated(self, instance):
        self.add_log(
            f"Parser {instance.parser_name} has started saving the translated data"
        )

    def on_finish_save_translated(self, instance):
        self.add_log(
            f"Parser {instance.parser_name} has finished saving the translated data"
        )
