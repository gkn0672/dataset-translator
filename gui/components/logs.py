import gradio as gr


def create_logs():
    """
    Create the logs output component.

    Returns:
        dict: Dictionary of log components
    """
    components = {}

    components["logs_output"] = gr.Textbox(
        label="Log Output",
        lines=18,
        max_lines=18,
        interactive=False,
        autoscroll=True,
        elem_id="logs-output",
    )

    return components
