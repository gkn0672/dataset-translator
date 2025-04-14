import gradio as gr


def create_source_config():
    """
    Create the source configuration components.

    Returns:
        dict: Dictionary of source configuration components
    """
    components = {}

    # State variables
    components["available_properties_state"] = gr.State([])

    with gr.Row():
        # Data source panel
        with gr.Column():
            with gr.Group():
                gr.Markdown("### 📂 Data Source")
                gr.HTML("")

                components["data_source_type"] = gr.Radio(
                    label="Source Type",
                    choices=["dataset", "file"],
                    value="dataset",
                    container=False,
                    interactive=True,
                )

                gr.HTML("")

                with gr.Row():
                    components["dataset_name"] = gr.Textbox(
                        label="Dataset Name",
                        placeholder="e.g., argilla/magpie-ultra-v0.1",
                        value="argilla/magpie-ultra-v0.1",
                        visible=True,
                        container=True,
                    )

                    components["file_path"] = gr.Textbox(
                        label="File Path",
                        placeholder="e.g., ./data/my_dataset.csv",
                        visible=False,
                        container=True,
                    )

                with gr.Row():
                    components["fetch_properties_btn"] = gr.Button(
                        "Fetch Properties",
                        variant="primary",
                        size="sm",
                        elem_id="fetch-btn",
                    )

        # Target config panel
        with gr.Column():
            with gr.Group():
                gr.Markdown("### 🎯 Target Configuration")

                components["target_config"] = gr.Dropdown(
                    label="Config Type",
                    choices=["QAConfig", "COTConfig"],
                    value="QAConfig",
                    container=True,
                )

                # Display available properties
                gr.Markdown("### ✅ Available Dataset Properties")
                components["available_properties_display"] = gr.Textbox(
                    label="",
                    placeholder="Properties will appear here after fetching",
                    interactive=False,
                    lines=3,
                    elem_id="properties-display",
                )

    return components
