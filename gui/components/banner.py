import gradio as gr
import os


def create_banner():
    """
    Create the banner component.

    Returns:
        dict: Dictionary of banner components
    """
    components = {}

    banner_path = os.environ.get("LOGO_URL")
    gr.HTML(f"""
        <div style="width: 100%;">
            <div style="width: 100%; margin-bottom: 10px;">
                <img src="{banner_path}" alt="Banner" style="width: 100%; height: 160px; object-fit: cover; border-radius: 8px 8px 0 0;">
            </div>
            <div style="background-color: rgba(31, 31, 31, 0.95); padding: 10px 20px; border-radius: 0 0 8px 8px; border-top: 2px solid #ff7f00; text-align: center;">
                <h1 style="margin: 0; font-size: 24px; font-weight: bold; color: white;">Dataset Translator</h1>
                <p style="margin-top: 5px; color: #cccccc;">Translate datasets from English to Vietnamese with custom field mappings</p>
                <p style="margin-top: 5px; color: #cccccc;">Developed by <a href="https://github.com/gkn0672" style="color: #ff7f00; text-decoration: none; font-weight: bold;">AGU</a></p>
            </div>
        </div>
    """)

    return components
