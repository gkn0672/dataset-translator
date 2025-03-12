import os
import re
import sys
import json
from typing import Union, List
from pydantic import Field
from openai import OpenAI

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
    Engine implementation for Ollama local LLM using the OpenAI Python library.
    """

    def __init__(self, model_name="llama3", host="http://localhost:11434"):
        super().__init__()
        self.model_name = model_name
        self.host = host

        # Initialize OpenAI client for Ollama
        self.client = OpenAI(
            base_url=f"{self.host}/v1",
            api_key="ollama",  # required but unused
        )

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

    @throttle(
        calls_per_minute=20,
        verbose=False,
        break_interval=1200,
        break_duration=60,
        jitter=3,
    )
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
                return [fail_translation_code, fail_translation_code]

            # Clear the cache if it's too large
            if len(CACHE_FAIL_PROMPT) > 10000:
                _, CACHE_FAIL_PROMPT = pop_half_dict(CACHE_FAIL_PROMPT)

            try:
                # Create messages for the chat API
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]

                # Call the OpenAI client with Ollama
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    top_p=0.4,
                    response_format={"type": "json_object"},
                )

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
                        return [fail_translation_code, fail_translation_code]
                    else:
                        CACHE_FAIL_PROMPT[input_hash] += 1
                else:
                    CACHE_FAIL_PROMPT[input_hash] = 1

                print(f"\nCurrent ollama fail cache: {CACHE_FAIL_PROMPT}\n")
                raise e

            try:
                # Parse the JSON response
                output_text = response.choices[0].message.content
                # Validate and parse the JSON
                output_schema = Translation.model_validate_json(output_text)
                output_dict = output_schema.model_dump()
                final_result = [
                    output_dict[f"translation_{i}"] for i in range(len(input_data))
                ]
            except Exception as json_error:
                print(f"Error parsing JSON response: {json_error}")
                return [fail_translation_code, fail_translation_code]

        else:
            # Single string translation
            system_prompt = (
                "You are a skilled translator tasked with translating text from **English** to **Vietnamese**. "
                "Avoid translating names, places, code snippets, LaTeX, and key phrases. "
                "Prioritize context to ensure an accurate and natural translation. "
                "Respond with **only the translation**, as it will be used directly."
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
                # Create messages for the chat API
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]

                # Call the OpenAI client with Ollama
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, temperature=0.3, top_p=0.4
                )

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

            # Extract translation from response
            final_result = response.choices[0].message.content

            # Clean the translation output
            final_result = final_result.replace(prefix_separator, "").replace(
                postfix_separator, ""
            )
            final_result = final_result.replace(prefix_prompt_block, "").replace(
                postfix_prompt_block, ""
            )
            final_result = self.remove_custom_brackets(final_result).strip()

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
                        final_result = [fail_translation_code, fail_translation_code]
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
                return [fail_translation_code, fail_translation_code]
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
