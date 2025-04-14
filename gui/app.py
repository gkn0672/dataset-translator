import gradio as gr
from gui.config.styles import CSS
from gui.components.banner import create_banner
from gui.components.logs import create_logs
from gui.components.source_config import create_source_config
from gui.components.field_mapping import create_field_mapping
from gui.components.translation_settings import create_translation_settings
from gui.handlers.common import register_all_handlers


def create_interface():
    """
    Create the Gradio interface with all components.

    Returns:
        gr.Blocks: The complete Gradio application
    """
    with gr.Blocks(title="Dataset Translator", css=CSS) as app:
        # Create components and store them in a dictionary
        components = {}

        # Top row with banner and logs
        with gr.Row():
            with gr.Column(scale=3):
                # Banner component
                banner_components = create_banner()
                components.update(banner_components)

            with gr.Column(scale=2):
                # Logs component
                log_components = create_logs()
                components.update(log_components)

        # Main tabs
        with gr.Row():
            with gr.Tabs():
                # Source & Config Tab
                with gr.TabItem("Source & Config", id="source_tab"):
                    source_components = create_source_config()
                    components.update(source_components)

                # Field Mapping Tab
                with gr.TabItem("Field Mapping", id="mapping_tab"):
                    mapping_components = create_field_mapping()
                    components.update(mapping_components)

                # Translation Settings Tab
                with gr.TabItem("Translation Settings", id="settings_tab"):
                    settings_components = create_translation_settings()
                    components.update(settings_components)

        # Start translation button (outside tabs)
        with gr.Row():
            components["submit_button"] = gr.Button(
                "🚀 Start Translation",
                variant="primary",
                size="lg",
                elem_id="start-translation-btn",
            )

        # Register all event handlers
        register_all_handlers(components)

    return app
