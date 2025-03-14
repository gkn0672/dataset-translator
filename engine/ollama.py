import re
import sys
import json
import requests
from typing import Union, List
from pydantic import Field

sys.path.insert(0, r"./")
from .base import BaseEngine
from .utils import hash_input, pop_half_dict, create_dynamic_model, throttle
from strings import remove_fuzzy_repeating_suffix

# Cache the fail prompt to avoid running translation again for subsequent calls
CACHE_FAIL_PROMPT = {}
MAX_LIST_RETRIES = 6  # The maximum number of retries for ollama list translation
MAX_STRING_RETRIES = 3  # The maximum number of retries for ollama string translation

# Cache the init prompt to avoid running translation again for subsequent calls
CACHE_INIT_PROMPT = {}

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
        super().__init__()
        self.model_name = model_name
        self.host = host

        # Use Ollama's native API endpoint
        self.api_url = f"{self.host}/api/chat"

        # Use a session for connection pooling
        self.session = requests.Session()

        self.translator = self  # Assign self as translator

    @staticmethod
    def construct_schema_prompt(schema: dict) -> str:
        schema_prompt = "Please provide the JSON object with the following schema:\n"

        json_prompt = json.dumps(
            {key: value["description"] for key, value in schema.items()}, indent=2
        )

        return schema_prompt + json_prompt

    @staticmethod
    def remove_custom_brackets(text: str) -> str:
        """
        Remove leading and trailing custom bracketed expressions from a given text.
        Custom brackets are defined as {|[|{ and }|]|}.

        Args:
            text (str): The input string from which custom bracketed expressions should be removed.

        Returns:
            str: The text with leading and trailing custom bracketed expressions removed.
        """
        pattern = r"^\s*\{\|\[\|\{.*?\}\|\]\|\}\s*|\s*\{\|\[\|\{.*?\}\|\]\|\}\s*$"
        return re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)

    def _do_translate(
        self,
        input_data: Union[str, List[str]],
        src: str,
        dest: str,
        fail_translation_code: str = "P1OP1_F",  # Pass in this code to replace the input_data if the exception is *unavoidable*
        **kwargs,
    ) -> Union[str, List[str]]:
        global CACHE_FAIL_PROMPT
        data_type = "list" if isinstance(input_data, list) else "str"

        if data_type == "list":
            translation_fields = {}
            prompt = ""
            for i in range(len(input_data)):
                translation_fields[f"translation_{i}"] = (
                    str,
                    Field(..., description=f"The translated text for text_{i}"),
                )
                prompt += (
                    "-" * 10 + f"\n text_{i}: {input_data[i]}\n" + "-" * 10
                    if len(input_data) > 1
                    else f"text_{i}: {input_data[i]}\n"
                )

            Translation = create_dynamic_model("Translation", translation_fields)

            system_prompt = (
                "You are a skilled translator tasked with converting text from **English** to **Vietnamese**. "
                "Be mindful not to translate specific items such as names, locations, code snippets, LaTeX, or key phrases. "
                "Ensure the translation reflects the context for accuracy and natural fluency. "
                "Your response must consist **only of the translated text** in JSON format."
            )
            postfix_system_prompt = f"{self.construct_schema_prompt(Translation.model_json_schema()['properties'])}"
            system_content = system_prompt + "\n\n" + postfix_system_prompt

            prefix_prompt_block = "{|[|{START_TRANSLATION_BLOCK}|]|}"
            postfix_prompt_block = "{|[|{END_TRANSLATION_BLOCK}|]|}"
            prefix_separator = "=" * 10
            postfix_separator = "=" * 10

            prefix_prompt = f"{prefix_prompt_block}\n"
            prefix_prompt += prefix_separator
            postfix_prompt_text = postfix_separator
            postfix_prompt_text += f"\n{postfix_prompt_block}"

            user_content = (
                prefix_prompt
                + "\n\n"
                + prompt
                + "\n\n"
                + postfix_prompt_text
                + "\n\n"
                + "Translate the provided text from **English** to **Vietnamese**, "
                + "considering the context. DO NOT add extra information or remove any information inside the fields. "
                + "Return the translated results in the respective fields of the JSON object."
            )

            # Calculate approximate token count (rough estimate)
            total_tokens = len((system_content + user_content).split())
            if total_tokens > 8000:
                return [fail_translation_code] * len(input_data)

            # Clear the cache if it's too large
            if len(CACHE_FAIL_PROMPT) > 10000:
                _, CACHE_FAIL_PROMPT = pop_half_dict(CACHE_FAIL_PROMPT)

            try:
                # Format messages for the native API
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]

                # Prepare payload for Ollama's native API
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "options": {"temperature": 0.3, "top_p": 0.4, "num_predict": 1024},
                    "stream": False,  # No streaming for translation
                }

                # Send request to Ollama's native API
                response = self.session.post(self.api_url, json=payload, timeout=60)

                response.raise_for_status()
                response_data = response.json()

                if "message" not in response_data:
                    raise ValueError("Invalid response format from Ollama API")

                if hash_input(input_data) in CACHE_FAIL_PROMPT:
                    CACHE_FAIL_PROMPT.pop(hash_input(input_data))

            except Exception as e:
                # Handle unavoidable exceptions
                input_hash = hash_input(input_data)

                if input_hash in CACHE_FAIL_PROMPT:
                    if CACHE_FAIL_PROMPT[input_hash] >= MAX_LIST_RETRIES:
                        print(
                            f"\nUnavoidable exception: {e}\nOllama max retries reached for list translation"
                        )
                        return [fail_translation_code] * len(input_data)
                    else:
                        CACHE_FAIL_PROMPT[input_hash] += 1
                else:
                    CACHE_FAIL_PROMPT[input_hash] = 1

                print(f"\nCurrent ollama fail cache: {CACHE_FAIL_PROMPT}\n")
                raise e

            try:
                # Extract content from Ollama's native API response
                output_text = response_data["message"]["content"]
                print(f"Response content (first 200 chars): {output_text[:200]}...")

                # Try multiple strategies to extract translations
                try:
                    # First strategy: clean up and parse JSON
                    clean_text = output_text.strip()
                    if clean_text.startswith("```json"):
                        clean_text = clean_text[7:]
                    if clean_text.endswith("```"):
                        clean_text = clean_text[:-3]
                    clean_text = clean_text.strip()

                    # Try to find a valid JSON block using regex
                    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
                    json_matches = re.findall(json_pattern, clean_text)

                    if json_matches:
                        print(f"Found potential JSON objects: {len(json_matches)}")
                        json_text = json_matches[0]
                        json_data = json.loads(json_text)
                    else:
                        # Fallback to direct parsing
                        json_data = json.loads(clean_text)

                    # Extract translations
                    final_result = []
                    for i in range(len(input_data)):
                        key = f"translation_{i}"
                        if key in json_data:
                            final_result.append(json_data[key])
                        else:
                            # Try alternative keys
                            for alt_key in [f"text_{i}", f"translated_{i}"]:
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
                        final_result = [fail_translation_code] * len(input_data)
            except Exception as e:
                print(f"Error processing response: {e}")
                return [fail_translation_code] * len(input_data)

        else:
            # Single string translation - simplified approach
            system_prompt = (
                "You are a skilled translator. Translate the following text from English to Vietnamese. "
                "Keep names, places, code snippets, LaTeX, and key technical terms unchanged. "
                "Return ONLY the translated text with no explanations, no formatting, and no additional content."
            )

            prefix_prompt_block = "{|[|{START_TRANSLATION_BLOCK}|]|}"
            postfix_prompt_block = "{|[|{END_TRANSLATION_BLOCK}|]|}"
            prefix_separator = "=" * 10
            postfix_separator = "=" * 10

            user_content = (
                f"{prefix_prompt_block}\n{prefix_separator}\n\n"
                f"{input_data}\n\n"
                f"{postfix_separator}\n{postfix_prompt_block}\n\n"
                "Translate all the above text inside the translation block from **English** to **Vietnamese**. "
                "DO NOT add extra information or remove any information inside, just translate."
            )

            # Approximate token count
            total_tokens = len((system_prompt + user_content).split())
            if total_tokens > 8000:
                return fail_translation_code

            # Clear the cache if it's too large
            if len(CACHE_FAIL_PROMPT) > 10000:
                _, CACHE_FAIL_PROMPT = pop_half_dict(CACHE_FAIL_PROMPT)

            try:
                # Format messages for Ollama's native API
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]

                # Prepare payload for Ollama's native API
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "options": {"temperature": 0.3, "top_p": 0.4, "num_predict": 1024},
                    "stream": False,  # No streaming for translation
                }

                # Send request to Ollama's native API
                response = self.session.post(self.api_url, json=payload, timeout=60)

                response.raise_for_status()
                response_data = response.json()

                if "message" not in response_data:
                    print(f"Unexpected response format: {response_data}")
                    raise ValueError("Invalid response format from Ollama API")

                if hash_input(input_data) in CACHE_FAIL_PROMPT:
                    CACHE_FAIL_PROMPT.pop(hash_input(input_data))

            except Exception as e:
                # Handle unavoidable exceptions
                input_hash = hash_input(input_data)

                if input_hash in CACHE_FAIL_PROMPT:
                    if CACHE_FAIL_PROMPT[input_hash] >= MAX_STRING_RETRIES:
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

            # Extract translation from response and log it
            content = response_data["message"]["content"]

            # Clean the translation output thoroughly
            final_result = content.replace(prefix_separator, "").replace(
                postfix_separator, ""
            )
            final_result = final_result.replace(prefix_prompt_block, "").replace(
                postfix_prompt_block, ""
            )
            final_result = self.remove_custom_brackets(final_result).strip()

            # Remove any markdown formatting that might be present
            final_result = re.sub(r"```.*?\n", "", final_result)
            final_result = re.sub(r"```", "", final_result)

            # Remove any "Translation:" prefix the model might add
            final_result = re.sub(
                r"^(Translation|Translated text|Vietnamese):\s*",
                "",
                final_result,
                flags=re.IGNORECASE,
            )

        # Process for repeating suffixes
        try:
            if data_type == "list":
                cleaned_output = []
                for data in final_result:
                    # Clean the translation output if there is any repeating suffix
                    output, percentage_removed = remove_fuzzy_repeating_suffix(
                        data, 0.8
                    )
                    if percentage_removed > SUFFIXES_PERCENTAGE and STRICT_TRANSLATION:
                        final_result = [fail_translation_code] * len(input_data)
                        break
                    else:
                        cleaned_output.append(
                            data
                        ) if KEEP_ORG_TRANSLATION else cleaned_output.append(output)
                final_result = cleaned_output
            else:
                output, percentage_removed = remove_fuzzy_repeating_suffix(
                    final_result, 0.8
                )
                if percentage_removed > SUFFIXES_PERCENTAGE and STRICT_TRANSLATION:
                    final_result = fail_translation_code
                else:
                    final_result = final_result if KEEP_ORG_TRANSLATION else output

        except Exception as e:
            print(f"\nError in cleaning the translation output: {e}\n")
            if data_type == "list":
                return [fail_translation_code] * len(input_data)
            return fail_translation_code

        return final_result


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
