import gradio as gr


def create_credential_manager():
    """
    Create the credential manager components.

    Returns:
        dict: Dictionary of credential manager components
    """
    components = {}

    with gr.Row():
        # Credentials configuration
        with gr.Column():
            with gr.Group():
                gr.HTML("🔑 API Credentials")

                components["groq_api_key"] = gr.Textbox(
                    label="Groq API Key",
                    placeholder="Enter your Groq API key",
                    value="",
                    container=True,
                    type="password",
                )

                components["hf_username"] = gr.Textbox(
                    label="Huggingface User/Organization",
                    placeholder="Username or organization name",
                    value="Terryagu",
                    container=True,
                )

                components["hf_token"] = gr.Textbox(
                    label="Huggingface Token",
                    placeholder="Enter your Huggingface token",
                    value="",
                    container=True,
                    type="password",
                )

    return components
