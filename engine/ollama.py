import re
import sys
import json
import requests
from typing import Union, List

sys.path.insert(0, r"./")
from .base import BaseEngine
from .utils import hash_input, pop_half_dict
from strings import clean_chinese_mix

# Cache the fail prompt to avoid running translation again for subsequent calls
CACHE_FAIL_PROMPT = {}
MAX_LIST_RETRIES = 6  # The maximum number of retries for ollama list translation
MAX_STRING_RETRIES = 3  # The maximum number of retries for ollama string translation

# If set to True, the translation will fail if the translation output contains repeating suffixes
# If set to False, the translation output will be cleaned and repeating suffixes will be removed
STRICT_TRANSLATION = True

# The percentage of the suffixes that should be repeating to be considered as a fail translation
SUFFIXES_PERCENTAGE = 20

# If set to True, the translation output will be kept if the translation output contains
# repeating suffixes but the percentage of the repeating suffixes is less than SUFFIXES_PERCENTAGE
KEEP_ORG_TRANSLATION = True


class OllamaEngine(BaseEngine):
    """
    Engine implementation for Ollama local LLM using Ollama's native API.
    """

    def __init__(self, model_name="llama3", host="http://localhost:11434"):
        """
        Initialize the Ollama Engine.

        Args:
            model_name (str): The name of the Ollama model to use
            host (str): The host URL where Ollama is running
        """
        super().__init__()
        self.model_name = model_name
        self.host = host

        # Use Ollama's native API endpoint
        self.api_url = f"{self.host}/api/chat"

        # Use a session for connection pooling
        self.session = requests.Session()

        self.translator = self  # Assign self as translator

    def _do_translate(
        self,
        input_data: Union[str, List[str]],
        src: str,
        dest: str,
        fail_translation_code: str = "P1OP1_F",
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Perform translation using Ollama API.

        Args:
            input_data: Text to translate (string or list of strings)
            src: Source language code
            dest: Destination language code
            fail_translation_code: Code to return if translation fails

        Returns:
            Translated text (string or list of strings)
        """
        global CACHE_FAIL_PROMPT
        data_type = "list" if isinstance(input_data, list) else "str"

        # Handle input data size limits
        if data_type == "list":
            text_length = sum(len(text) for text in input_data)
            if text_length > 8000:
                return [fail_translation_code] * len(input_data)
        elif len(input_data) > 8000:
            return fail_translation_code

        # Clear cache if too large
        if len(CACHE_FAIL_PROMPT) > 10000:
            _, CACHE_FAIL_PROMPT = pop_half_dict(CACHE_FAIL_PROMPT)

        try:
            if data_type == "list":
                # Create prompt for list translation
                system_prompt = (
                    "You are a professional English to Vietnamese translator. Translate accurately while preserving names, "
                    "locations, code, and technical terms. Respond with only valid JSON in this exact format:"
                )

                # Create JSON schema example
                json_example = "{\n"
                for i in range(len(input_data)):
                    json_example += (
                        f'  "translation_{i}": "Vietnamese translation of text_{i}",\n'
                    )
                json_example = json_example.rstrip(",\n") + "\n}"

                system_prompt += f"\n\n{json_example}"

                # Create user prompt with texts to translate
                user_prompt = "Translate these texts from English to Vietnamese:\n\n"
                for i, text in enumerate(input_data):
                    user_prompt += f"text_{i}: {text}\n\n"

                user_prompt += "Return only a valid JSON object with the translations."
            else:
                # Create prompt for single text translation
                system_prompt = (
                    "You are a professional English to Vietnamese translator. Translate accurately while preserving names, "
                    "locations, code, and technical terms. Respond with ONLY the translation, nothing else."
                )

                # Create user prompt with text to translate
                user_prompt = (
                    f"Translate this text from English to Vietnamese:\n\n{input_data}"
                )

            # Format messages for Ollama API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Prepare API payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.4,
                    "num_predict": 4096,
                    "context_window": 8000,
                },
                "stream": False,
            }

            # Send request
            response = self.session.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()

            if "message" not in response_data:
                raise ValueError("Invalid response format from Ollama API")

            # Process response
            output_text = response_data["message"]["content"]

            # Remove hash from fail cache if request succeeded
            if hash_input(input_data) in CACHE_FAIL_PROMPT:
                CACHE_FAIL_PROMPT.pop(hash_input(input_data))

            # Process the response based on data type
            if data_type == "list":
                # Extract JSON from response (handling potential code blocks)
                clean_text = output_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()

                try:
                    # Try to parse the json
                    json_data = json.loads(clean_text)

                    # Extract translations
                    final_result = []
                    for i in range(len(input_data)):
                        key = f"translation_{i}"
                        if key in json_data:
                            final_result.append(json_data[key])
                        else:
                            # Try alternative keys
                            alt_keys = [f"text_{i}", f"translated_{i}"]
                            for alt_key in alt_keys:
                                if alt_key in json_data:
                                    final_result.append(json_data[alt_key])
                                    break
                            else:
                                raise KeyError(f"Missing key {key} in JSON response")
                except Exception as json_error:
                    print(f"JSON parsing error: {json_error}")
                    # Try to extract translations using regex pattern
                    pattern = r'"translation_(\d+)"\s*:\s*"([^"]+)"'
                    matches = re.findall(pattern, output_text)

                    if matches:
                        print(
                            f"Extracting translations with regex. Found {len(matches)} matches."
                        )
                        translations = {}
                        for idx_str, text in matches:
                            idx = int(idx_str)
                            translations[idx] = text

                        final_result = []
                        for i in range(len(input_data)):
                            if i in translations:
                                final_result.append(translations[i])
                            else:
                                final_result.append(fail_translation_code)
                    else:
                        print("Could not extract translations. Using fail code.")
                        return [fail_translation_code] * len(input_data)
            else:
                # Clean single text response
                final_result = output_text.strip()

                # Remove any markdown formatting
                if final_result.startswith("```") and final_result.endswith("```"):
                    final_result = final_result[3:-3].strip()

                # Remove any prefixes the model might add
                prefixes = ["Translation:", "Translated text:", "Vietnamese:"]
                for prefix in prefixes:
                    if final_result.lower().startswith(prefix.lower()):
                        final_result = final_result[len(prefix) :].strip()

            # Check for repeating suffixes
            try:
                if data_type == "list":
                    cleaned_output = []
                    for data in final_result:
                        data = clean_chinese_mix(data)
                        cleaned_output.append(data if KEEP_ORG_TRANSLATION else data)
                    final_result = cleaned_output
                else:
                    final_result = clean_chinese_mix(final_result)
            except Exception as e:
                print(f"\nError in cleaning the translation output: {e}\n")
                return (
                    [fail_translation_code] * len(input_data)
                    if data_type == "list"
                    else fail_translation_code
                )

            return final_result

        except Exception as e:
            # Handle unavoidable exceptions
            input_hash = hash_input(input_data)

            # Track retry count
            if input_hash in CACHE_FAIL_PROMPT:
                if (
                    data_type == "list"
                    and CACHE_FAIL_PROMPT[input_hash] >= MAX_LIST_RETRIES
                ):
                    print(
                        f"\nUnavoidable exception: {e}\nOllama max retries reached for list translation"
                    )
                    return [fail_translation_code] * len(input_data)
                elif (
                    data_type == "str"
                    and CACHE_FAIL_PROMPT[input_hash] >= MAX_STRING_RETRIES
                ):
                    print(
                        f"\nUnavoidable exception: {e}\nOllama max retries reached for string translation"
                    )
                    return fail_translation_code
                else:
                    CACHE_FAIL_PROMPT[input_hash] += 1
            else:
                CACHE_FAIL_PROMPT[input_hash] = 1

            print(f"\nCurrent ollama fail cache: {CACHE_FAIL_PROMPT}\n")
            raise e

    def translate(self, input_data, src, dest, fail_translation_code="P1OP1_F"):
        """
        Calls the BaseEngine's translate method. This is just a wrapper for clarity.
        """
        return super().translate(input_data, src, dest, fail_translation_code)


if __name__ == "__main__":
    test = OllamaEngine()

    # Get the time taken
    import time

    start = time.time()
    print(test.translate(["Hello", "How are you today?"], src="en", dest="vi"))
    print(test.translate("Hello", src="en", dest="vi"))
    print(f"Time taken: {time.time() - start}")

    start = time.time()
    print(test.translate(["VIETNAMESE", "JAPANESE"], src="en", dest="vi"))
    print(test.translate("HELLO IN VIETNAMESE", src="en", dest="vi"))
    print(f"Time taken: {time.time() - start}")
