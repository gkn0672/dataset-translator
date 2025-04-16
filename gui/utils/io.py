import io
import sys
import threading
import time
import re


class StdoutCapture:
    """Class to capture standard output and error for logging purposes."""

    def __init__(self, callback_fn=None):
        """
        Initialize the stdout capture buffer with optional callback function.

        Args:
            callback_fn: Optional function to call with captured content for real-time updates
        """
        self.buffer = io.StringIO()
        self.old_stdout = None
        self.old_stderr = None
        self.callback_fn = callback_fn
        self.running = False
        self.update_thread = None
        self.lock = threading.Lock()
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def start(self):
        """Start capturing stdout and stderr."""
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.buffer
        sys.stderr = self.buffer

        if self.callback_fn:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_callback_loop)
            self.update_thread.daemon = True
            self.update_thread.start()

    def stop(self):
        """Stop capturing stdout and restore original."""
        if self.old_stdout:
            sys.stdout = self.old_stdout
        if self.old_stderr:
            sys.stderr = self.old_stderr

        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)

    def get_content(self):
        """Get the captured content as a string."""
        with self.lock:
            content = self.buffer.getvalue()
            # Remove ANSI escape sequences for cleaner output
            return self.ansi_escape.sub("", content)

    def _update_callback_loop(self):
        """Periodically check buffer for new content and call callback."""
        last_position = 0
        while self.running:
            with self.lock:
                current_value = self.buffer.getvalue()
                if len(current_value) > last_position:
                    # Update our position in the buffer
                    last_position = len(current_value)
                    # Remove ANSI escape sequences for cleaner display
                    cleaned_value = self.ansi_escape.sub("", current_value)
                    # Replace carriage returns with newlines to handle progress bars
                    display_value = cleaned_value.replace("\r", "\n")
                    # Send the cleaned log content to the callback
                    self.callback_fn(display_value)
            time.sleep(0.1)  # Update frequency


def render_additional_fields(additional_fields):
    """
    Render additional fields as an HTML table.

    Args:
        additional_fields (list): List of field dictionaries with "key" and "value"

    Returns:
        str: HTML representation of the fields table
    """
    if not additional_fields:
        return """
            No additional fields added yet
            """

    html = """
        
        
            Field Key
            Field Value
        """

    for field in additional_fields:
        html += f"""
            {field["key"]}
            {field["value"]}
        """

    html += ""
    return html
