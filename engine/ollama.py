import requests
from typing import Union, List
from .base import BaseEngine


class OllamaEngine(BaseEngine):
    """
    Engine implementation for Ollama local LLM.
    """

    def __init__(self, model_name="llama2", host="http://localhost:11434"):
        super().__init__()
        self.model_name = model_name
        self.host = host
        self.translator = self  # Assign self as translator
        self.api_endpoint = f"{self.host}/api/generate"

    def _do_translate(
        self,
        input_data: Union[str, List[str]],
        src: str,
        dest: str,
        fail_translation_code: str = "P1OP1_F",
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Perform translation using Ollama.

        Args:
            input_data (Union[str, List[str]]): The input data to be translated.
            src (str): The source language code.
            dest (str): The destination language code.
            fail_translation_code (str, optional): The code to be returned when translation fails.
            **kwargs: Additional keyword arguments for translation.

        Returns:
            Union[str, List[str]]: The translated output data.
        """
        if src != "en" or dest != "vi":
            raise ValueError(
                "This Engine only supports English to Vietnamese translation"
            )

        if isinstance(input_data, list):
            return [
                self._translate_text(text, fail_translation_code) for text in input_data
            ]
        else:
            return self._translate_text(input_data, fail_translation_code)

    def _translate_text(self, text: str, fail_translation_code: str) -> str:
        """
        Translate a single text using Ollama.

        Args:
            text (str): The text to be translated.
            fail_translation_code (str): The code to be returned when translation fails.

        Returns:
            str: The translated text.
        """
        try:
            prompt = f"Translate the following English text to Vietnamese: \n\n{text}"

            payload = {"model": self.model_name, "prompt": prompt, "stream": False}

            response = requests.post(self.api_endpoint, json=payload)

            if response.status_code == 200:
                result = response.json()
                return result.get("response", fail_translation_code)
            else:
                return fail_translation_code
        except Exception as e:
            print(f"Translation error: {e}")
            return fail_translation_code
