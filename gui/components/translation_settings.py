import gradio as gr


def create_translation_settings():
    """
    Create the translation settings components.

    Returns:
        dict: Dictionary of translation settings components
    """
    components = {}

    with gr.Row():
        # Translator configuration
        with gr.Column():
            with gr.Group():
                gr.HTML("🤖 Translator Configuration")

                components["translator_engine"] = gr.Dropdown(
                    label="Translator Engine",
                    choices=["ollama", "groq"],
                    value="ollama",
                    container=True,
                )

                components["translator_model"] = gr.Textbox(
                    label="Model Name",
                    placeholder="e.g., llama3.1:8b-instruct-q4_0",
                    value="llama3.1:8b-instruct-q4_0",
                    container=True,
                )

                with gr.Row():
                    with gr.Column():
                        components["use_verbose"] = gr.Checkbox(
                            label="Verbose Logging",
                            value=True,
                            container=True,
                        )

                    with gr.Column():
                        components["push_to_huggingface"] = gr.Checkbox(
                            label="Push to HuggingFace",
                            value=False,
                            container=True,
                        )

        # Performance settings
        with gr.Column():
            with gr.Group():
                gr.HTML("⚙️ Performance Settings")

                components["limit"] = gr.Number(
                    label="Limit (records)",
                    value=10,
                    precision=0,
                    container=True,
                )

                components["max_memory_percent"] = gr.Slider(
                    label="Max Memory %",
                    minimum=0.1,
                    maximum=0.9,
                    value=0.6,
                    step=0.1,
                    container=True,
                )

                with gr.Row():
                    with gr.Column():
                        components["min_batch_size"] = gr.Number(
                            label="Min Batch Size",
                            value=1,
                            precision=0,
                            container=True,
                        )

                    with gr.Column():
                        components["max_batch_size"] = gr.Number(
                            label="Max Batch Size",
                            value=5,
                            precision=0,
                            container=True,
                        )

    return components
