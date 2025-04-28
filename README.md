<div align="center">
<img src="https://github.com/bloomifycafe/blossomsAI/blob/main/assets/logo.png?raw=true" alt="Logo"/>
</div>
</br>
<div align="center">

# 📚 Dataset Translator 

</div>

## 🌟 Introduction

Dataset Translator is a powerful tool designed to translate datasets from English to Vietnamese. It provides a user-friendly GUI built with Gradio for translating various data formats including JSON and Parquet files. The tool supports multiple translation engines including:

- **Ollama**: For local LLM-based translation using models like Llama
- **Groq**: For cloud-based translation using Groq's API

The application handles various dataset formats and can be used either through its graphical interface or programmatically. It's designed to maintain the structure of the original dataset while accurately translating text fields.

Key features include:
- 🔄 Support for various data formats (JSON, JSONL, Parquet)
- 🌐 Integration with Hugging Face datasets
- 🧠 Multiple translation engine options
- 📊 Smart memory management for large datasets
- 🔁 Field mapping for flexible configuration
- ⚙️ Batched processing for performance optimization
- 📤 Optional upload to Hugging Face Hub

## 💻 Installation

### Prerequisites

- Python 3.10 or higher
- uv package manager ([uv.install](https://github.com/astral-sh/uv))

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/dataset-translator.git
cd dataset-translator
```

### Step 2: Create and activate a virtual environment

```bash
# Create a virtual environment
uv venv create

# Activate the virtual environment
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
uv install
```

## ⚙️ Configuration

_Configuration details will be added soon._

## 🚀 Running the Application

After installation and configuration, you can run the application with:

```bash
python main.py
```

This will launch the Gradio web interface that you can access through your browser.