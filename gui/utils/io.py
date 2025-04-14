import io
import sys


class StdoutCapture:
    """Class to capture standard output for logging purposes."""

    def __init__(self):
        """Initialize the stdout capture buffer."""
        self.buffer = io.StringIO()
        self.old_stdout = None

    def start(self):
        """Start capturing stdout."""
        self.old_stdout = sys.stdout
        sys.stdout = self.buffer

    def stop(self):
        """Stop capturing stdout and restore original."""
        if self.old_stdout:
            sys.stdout = self.old_stdout

    def get_content(self):
        """Get the captured content as a string."""
        return self.buffer.getvalue()


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
