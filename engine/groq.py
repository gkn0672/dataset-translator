import os
import re
import sys
import json
from typing import Union, List

from pydantic import Field

sys.path.insert(0, r"./")
from groq import Groq
from .base import BaseEngine
from .utils import hash_input, pop_half_dict, create_dynamic_model, throttle
from strings import remove_fuzzy_repeating_suffix

# Cache the fail prompt to avoid running translation again for subsequent calls
CACHE_FAIL_PROMPT = {}
MAX_LIST_RETRIES = 6  # The maximum number of retries for groq list translation
MAX_STRING_RETRIES = 3  # The maximum number of retries for groq string translation


# Cache the init prompt to avoid running translation again for subsequent calls
CACHE_INIT_PROMPT = {}

# If set to True, the translation will fail if the translation output contains repeating suffixes, if set to False, the translation output will be cleaned and the repeating suffixes will be removed
STRICT_TRANSLATION = True

# The percentage of the suffixes that should be repeating to be considered as a fail translation
SUFFIXES_PERCENTAGE = 20

# If set to True, the translation output will be kept if the translation output contains repeating suffixes but the percentage of the repeating suffixes is less than SUFFIXES_PERCENTAGE
KEEP_ORG_TRANSLATION = True


# The GroqEngine class is a Engine that uses the Groq API to translate text from one language to another via LLM, expect a high quality translation but it is very slow (100 examples every 6-7 minutes)
# The list translation is supported, the Engine will return a list of translated text but Groq API for JSON is very slow and unreliable, so it is not recommended to use this Engine for list translation
class GroqEngine(BaseEngine):
    def __init__(self):
        try:
            self.groq_client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
        except KeyError:
            raise KeyError(
                "Please set the environment variable GROQ_API_KEY by running `export GROQ_API_KEY=<your_api_key>`, the API key can be obtained from https://console.groq.com/keys, it is free to sign up and use the API."
            )

        self.translator = self.groq_client.chat.completions.create

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
            postfix_prompt = (
                "Translate the provided text from **English** to **Vietnamese**, "
                "considering the context. DO NOT add extra information or remove any information inside the fields. Return the translated results in the respective fields of the JSON object."
            )

        else:
            system_prompt = (
                "You are a skilled translator tasked with translating text from **English** to **Vietnamese**. "
                "Avoid translating names, places, code snippets, LaTeX, and key phrases. "
                "Prioritize context to ensure an accurate and natural translation. "
                "Respond with **only the translation**, as it will be used directly."
            )
            postfix_system_prompt = ""
            prompt = input_data
            postfix_prompt = (
                "Translate all the above text inside the translation block from **English** to **Vietnamese**. "
                "DO NOT add extra information or remove any information inside, just translate."
            )

        prefix_prompt_block = "{|[|{START_TRANSLATION_BLOCK}|]|}"
        postfix_prompt_block = "{|[|{END_TRANSLATION_BLOCK}|]|}"
        prefix_separator = "=" * 10
        postfix_separator = "=" * 10

        prefix_prompt = f"{prefix_prompt_block}\n"
        prefix_prompt += prefix_separator
        postfix_prompt_text = postfix_separator
        postfix_prompt_text += f"\n{postfix_prompt_block}"

        final_system_prompt = system_prompt + (
            "\n\n" + postfix_system_prompt if postfix_system_prompt else ""
        )
        final_user_prompt = (
            prefix_prompt
            + "\n\n"
            + prompt
            + "\n\n"
            + postfix_prompt_text
            + "\n\n"
            + postfix_prompt
        )

        chat_args = {
            "messages": [
                {
                    "role": "system",
                    "content": final_system_prompt,
                },
                {"role": "user", "content": final_user_prompt},
            ],
            "model": "llama3-8b-8192",
            "temperature": 0.3,
            "top_p": 0.4,
            "max_tokens": 8000,
            "stream": False,
        }

        if data_type == "list":
            chat_args["response_format"] = {"type": "json_object"}

        if len((final_system_prompt + final_user_prompt).split()) > 8000:
            if data_type == "list":
                return [fail_translation_code, fail_translation_code]
            return fail_translation_code

        # Clear the cache if the cache is too large
        if len(CACHE_FAIL_PROMPT) > 10000:
            _, CACHE_FAIL_PROMPT = pop_half_dict(CACHE_FAIL_PROMPT)

        try:
            output = self.translator(**chat_args)
            if hash_input(input_data) in CACHE_FAIL_PROMPT:
                CACHE_FAIL_PROMPT.pop(hash_input(input_data))
        except Exception as e:
            # Check if the exception is unavoidable by matching the prompt with the cache fail prompt key
            input_hash = hash_input(input_data)

            if input_hash in CACHE_FAIL_PROMPT:
                if (
                    data_type == "list"
                    and CACHE_FAIL_PROMPT[input_hash] >= MAX_LIST_RETRIES
                ):
                    print(
                        f"\nUnavoidable exception: {e}\nGroq max retries reached for list translation"
                    )
                    return [fail_translation_code, fail_translation_code]
                elif (
                    data_type == "str"
                    and CACHE_FAIL_PROMPT[input_hash] >= MAX_STRING_RETRIES
                ):
                    print(
                        f"\nUnavoidable exception: {e}\nGroq max retries reached for string translation"
                    )
                    return fail_translation_code
                else:
                    CACHE_FAIL_PROMPT[input_hash] += 1
            else:
                CACHE_FAIL_PROMPT[input_hash] = 1

            print(f"\nCurrent groq fail cache: {CACHE_FAIL_PROMPT}\n")
            raise e

        if data_type == "list":
            output_text = output.choices[0].message.content
            output_schema = Translation.model_validate_json(output_text)
            output_dict = output_schema.model_dump()
            final_result = [
                output_dict[f"translation_{i}"] for i in range(len(input_data))
            ]
        else:
            final_result = output.choices[0].message.content

            # Clean the translation output if the model repeat the prefix and postfix prompt
            final_result = final_result.replace(prefix_separator, "").replace(
                postfix_separator, ""
            )
            final_result = final_result.replace(prefix_prompt_block, "").replace(
                postfix_prompt_block, ""
            )
            final_result = self.remove_custom_brackets(final_result).strip()

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
    test = GroqEngine()

    # Get the time taken
    import time

    start = time.time()
    print(test.translate(["Hello", "How are you today ?"], src="en", dest="vi"))
    print(test.translate("Hello", src="en", dest="vi"))
    print(f"Time taken: {time.time() - start}")

    start = time.time()
    print(test.translate(["VIETNAMESE", "JAPANSESE"], src="en", dest="vi"))
    print(test.translate("HELLO IN VIETNAMSE", src="en", dest="vi"))
    print(f"Time taken: {time.time() - start}")

    start = time.time()
    print(test.translate(["Hello", "How are you today ?"], src="en", dest="vi"))
    print(test.translate("Hello", src="en", dest="vi"))
    print(f"Time taken: {time.time() - start}")

    start = time.time()
    print(test.translate(["VIETNAMESE", "JAPANSESE"], src="en", dest="vi"))
    print(test.translate("HELLO IN VIETNAMSE", src="en", dest="vi"))
    print(f"Time taken: {time.time() - start}")

    start = time.time()
    print(
        test.translate(
            """Q:Information:  - The Assistant Secretary of Defense for Health Affairs (ASD(HA)) is chartered under United States Department of Defense Directive (DoDD) 5136.1 in 1994. This DoDD states that the ASD(HA) is the principal advisor to the U.S. Secretary of Defense on all "DoD health policies, programs and activities." In addition to exercising oversight of all DoD health resources, ASD(HA) serves as director of the Tricare Management Activity.  - The Department of the Air Force (DAF) is one of the three Military Departments within the Department of Defense of the United States of America. The Department of the Air Force was formed on September 18, 1947, per the National Security Act of 1947 and it includes all elements and units of the United States Air Force (USAF).  - The Surgeon General of the Air Force is the senior-most Medical Service officer in the United States Department of the Air Force. In recent times, this has been a Lieutenant General who serves as head of the United States Air Force Medical Service (AFMS). The Surgeon General is usually the senior Medical Corps officer, but acting surgeons general have been from other branches of the medical service.  - Lieutenant general, lieutenant-general and similar (abbrev Lt Gen, LTG and similar) is a three-star military rank (NATO code OF-8) used in many countries. The rank traces its origins to the Middle Ages, where the title of lieutenant general was held by the second in command on the battlefield, who was normally subordinate to a captain general.  - The United States Air Force (USAF) is the aerial warfare service branch of the United States Armed Forces and one of the seven American uniformed services. Initially part of the United States Army, the USAF was formed as a separate branch of the military on 18 September 1947 under the National Security Act of 1947. It is the most recent branch of the U.S. military to be formed, and is the largest and one of the world's most technologically advanced air forces. The USAF articulates its core functions as Nuclear Deterrence Operations, Special Operations, Air Superiority, Global Integrated ISR, Space Superiority, Command and Control, Cyberspace Superiority, Personnel Recovery, Global Precision Attack, Building Partnerships, Rapid Global Mobility and Agile Combat Support.  - Lieutenant General James Gordon Roudebush , USAF , ( born February 24 , 1948 ) was the 19th Surgeon General of the United States Air Force , Headquarters U.S. Air Force , Washington , D.C. General Roudebush served as functional manager of the U.S. Air Force Medical Service . In this capacity , he advised the Secretary of the Air Force and Air Force Chief of Staff , as well as the Assistant Secretary of Defense for Health Affairs on matters pertaining to the medical aspects of the air expeditionary force and the health of Air Force people . General Roudebush had authority to commit resources worldwide for the Air Force Medical Service , to make decisions affecting the delivery of medical services , and to develop plans , programs and procedures to support worldwide medical service missions . He exercised direction , guidance and technical management of more than 42,400 people assigned to 74 medical facilities worldwide . A native of Gering , Nebraska , Roudebush entered the Air Force in 1975 after receiving a Bachelor of Medicine degree from the University of Nebraska at Lincoln , and a Doctor of Medicine degree from the University of Nebraska College of Medicine . He completed residency training in family practice at the Wright - Patterson Air Force Medical Center , Ohio , in 1978 , and aerospace medicine at Brooks Air Force Base , Texas , in 1984 . He commanded a wing clinic and wing hospital before becoming Deputy Commander of the Air Force Materiel Command Human Systems Center . He has served as Command Surgeon for U.S. Central Command , Pacific Air Forces , U.S. Transportation Command and Headquarters Air Mobility Command . Prior to his selection as the 19th Surgeon General , he served as the Deputy Surgeon General of the U.S. Air Force . He retired from the U.S. Air Force on October 1 , 2009 .    After reading the paragraphs above, choose the best answer for the entity that related to 'james g. roudebush' with the relationship of 'occupation'.  Choices: - advisor  - army  - captain  - general  - lieutenant  - military  - officer  - secretary  - surgeon  - united states of america
A:""",
            src="en",
            dest="vi",
        )
    )
    print(f"Time taken: {time.time() - start}")

    start = time.time()
    print(
        test.translate(
            """Q:Information:  - The Assistant Secretary of Defense for Health Affairs (ASD(HA)) is chartered under United States Department of Defense Directive (DoDD) 5136.1 in 1994. This DoDD states that the ASD(HA) is the principal advisor to the U.S. Secretary of Defense on all "DoD health policies, programs and activities." In addition to exercising oversight of all DoD health resources, ASD(HA) serves as director of the Tricare Management Activity.  - The Department of the Air Force (DAF) is one of the three Military Departments within the Department of Defense of the United States of America. The Department of the Air Force was formed on September 18, 1947, per the National Security Act of 1947 and it includes all elements and units of the United States Air Force (USAF).  - The Surgeon General of the Air Force is the senior-most Medical Service officer in the United States Department of the Air Force. In recent times, this has been a Lieutenant General who serves as head of the United States Air Force Medical Service (AFMS). The Surgeon General is usually the senior Medical Corps officer, but acting surgeons general have been from other branches of the medical service.  - Lieutenant general, lieutenant-general and similar (abbrev Lt Gen, LTG and similar) is a three-star military rank (NATO code OF-8) used in many countries. The rank traces its origins to the Middle Ages, where the title of lieutenant general was held by the second in command on the battlefield, who was normally subordinate to a captain general.  - The United States Air Force (USAF) is the aerial warfare service branch of the United States Armed Forces and one of the seven American uniformed services. Initially part of the United States Army, the USAF was formed as a separate branch of the military on 18 September 1947 under the National Security Act of 1947. It is the most recent branch of the U.S. military to be formed, and is the largest and one of the world's most technologically advanced air forces. The USAF articulates its core functions as Nuclear Deterrence Operations, Special Operations, Air Superiority, Global Integrated ISR, Space Superiority, Command and Control, Cyberspace Superiority, Personnel Recovery, Global Precision Attack, Building Partnerships, Rapid Global Mobility and Agile Combat Support.  - Lieutenant General James Gordon Roudebush , USAF , ( born February 24 , 1948 ) was the 19th Surgeon General of the United States Air Force , Headquarters U.S. Air Force , Washington , D.C. General Roudebush served as functional manager of the U.S. Air Force Medical Service . In this capacity , he advised the Secretary of the Air Force and Air Force Chief of Staff , as well as the Assistant Secretary of Defense for Health Affairs on matters pertaining to the medical aspects of the air expeditionary force and the health of Air Force people . General Roudebush had authority to commit resources worldwide for the Air Force Medical Service , to make decisions affecting the delivery of medical services , and to develop plans , programs and procedures to support worldwide medical service missions . He exercised direction , guidance and technical management of more than 42,400 people assigned to 74 medical facilities worldwide . A native of Gering , Nebraska , Roudebush entered the Air Force in 1975 after receiving a Bachelor of Medicine degree from the University of Nebraska at Lincoln , and a Doctor of Medicine degree from the University of Nebraska College of Medicine . He completed residency training in family practice at the Wright - Patterson Air Force Medical Center , Ohio , in 1978 , and aerospace medicine at Brooks Air Force Base , Texas , in 1984 . He commanded a wing clinic and wing hospital before becoming Deputy Commander of the Air Force Materiel Command Human Systems Center . He has served as Command Surgeon for U.S. Central Command , Pacific Air Forces , U.S. Transportation Command and Headquarters Air Mobility Command . Prior to his selection as the 19th Surgeon General , he served as the Deputy Surgeon General of the U.S. Air Force . He retired from the U.S. Air Force on October 1 , 2009 .    After reading the paragraphs above, choose the best answer for the entity that related to 'james g. roudebush' with the relationship of 'occupation'.  Choices: - advisor  - army  - captain  - general  - lieutenant  - military  - officer  - secretary  - surgeon  - united states of america
A:""",
            src="en",
            dest="vi",
        )
    )
    print(f"Time taken: {time.time() - start}")

    start = time.time()
    print(
        test.translate(
            [
                """Q:Information:  - The Assistant Secretary of Defense for Health Affairs (ASD(HA)) is chartered under United States Department of Defense Directive (DoDD) 5136.1 in 1994. This DoDD states that the ASD(HA) is the principal advisor to the U.S. Secretary of Defense on all "DoD health policies, programs and activities." In addition to exercising oversight of all DoD health resources, ASD(HA) serves as director of the Tricare Management Activity.  - The Department of the Air Force (DAF) is one of the three Military Departments within the Department of Defense of the United States of America. The Department of the Air Force was formed on September 18, 1947, per the National Security Act of 1947 and it includes all elements and units of the United States Air Force (USAF).  - The Surgeon General of the Air Force is the senior-most Medical Service officer in the United States Department of the Air Force. In recent times, this has been a Lieutenant General who serves as head of the United States Air Force Medical Service (AFMS). The Surgeon General is usually the senior Medical Corps officer, but acting surgeons general have been from other branches of the medical service.  - Lieutenant general, lieutenant-general and similar (abbrev Lt Gen, LTG and similar) is a three-star military rank (NATO code OF-8) used in many countries. The rank traces its origins to the Middle Ages, where the title of lieutenant general was held by the second in command on the battlefield, who was normally subordinate to a captain general.  - The United States Air Force (USAF) is the aerial warfare service branch of the United States Armed Forces and one of the seven American uniformed services. Initially part of the United States Army, the USAF was formed as a separate branch of the military on 18 September 1947 under the National Security Act of 1947. It is the most recent branch of the U.S. military to be formed, and is the largest and one of the world's most technologically advanced air forces. The USAF articulates its core functions as Nuclear Deterrence Operations, Special Operations, Air Superiority, Global Integrated ISR, Space Superiority, Command and Control, Cyberspace Superiority, Personnel Recovery, Global Precision Attack, Building Partnerships, Rapid Global Mobility and Agile Combat Support.  - Lieutenant General James Gordon Roudebush , USAF , ( born February 24 , 1948 ) was the 19th Surgeon General of the United States Air Force , Headquarters U.S. Air Force , Washington , D.C. General Roudebush served as functional manager of the U.S. Air Force Medical Service . In this capacity , he advised the Secretary of the Air Force and Air Force Chief of Staff , as well as the Assistant Secretary of Defense for Health Affairs on matters pertaining to the medical aspects of the air expeditionary force and the health of Air Force people . General Roudebush had authority to commit resources worldwide for the Air Force Medical Service , to make decisions affecting the delivery of medical services , and to develop plans , programs and procedures to support worldwide medical service missions . He exercised direction , guidance and technical management of more than 42,400 people assigned to 74 medical facilities worldwide . A native of Gering , Nebraska , Roudebush entered the Air Force in 1975 after receiving a Bachelor of Medicine degree from the University of Nebraska at Lincoln , and a Doctor of Medicine degree from the University of Nebraska College of Medicine . He completed residency training in family practice at the Wright - Patterson Air Force Medical Center , Ohio , in 1978 , and aerospace medicine at Brooks Air Force Base , Texas , in 1984 . He commanded a wing clinic and wing hospital before becoming Deputy Commander of the Air Force Materiel Command Human Systems Center . He has served as Command Surgeon for U.S. Central Command , Pacific Air Forces , U.S. Transportation Command and Headquarters Air Mobility Command . Prior to his selection as the 19th Surgeon General , he served as the Deputy Surgeon General of the U.S. Air Force . He retired from the U.S. Air Force on October 1 , 2009 .    After reading the paragraphs above, choose the best answer for the entity that related to 'james g. roudebush' with the relationship of 'occupation'.  Choices: - advisor  - army  - captain  - general  - lieutenant  - military  - officer  - secretary  - surgeon  - united states of america
A:"""
            ],
            src="en",
            dest="vi",
        )
    )
    print(f"Time taken: {time.time() - start}")
