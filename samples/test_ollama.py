# TODO: experiment with data read with stream mode
import sys

sys.path.insert(0, r"./")
from tqdm.auto import tqdm

from datasets import load_dataset

from config.default import DefaultConfig
from translator.parser import DataParser
from translator.callback import VerboseCallback
from engine.ollama import OllamaEngine


PARSER_NAME = "MagpieUltraV01"


# Patience is the key since the data is large and is using an LLM based translator
class MagpieUltraV01Parser(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(
            file_path,
            output_path,
            parser_name=PARSER_NAME,
            target_config=DefaultConfig,  # The data config to be validated to check if self implement "convert" function is correct or not,
            # you must map the data form to the correct fields of the @dataclass in the configs/base_config.py
            target_fields=[
                "question_text",
                "orig_answer_texts",
            ],  # The data fields to be translated (The fields belong to BaseConfig)
            do_translate=True,
            no_translated_code=False,  # Remove any instance of string that appears to be coding language (e.g. Python code, HTML, etc.)
            translator=OllamaEngine(
                model_name="llama3.2:3b-instruct-q8_0"
            ),  # Groq is very slow but it is a high quality translator
            parser_callbacks=[
                VerboseCallback
            ],  # The callback to be called after the data has been converted and translated
            max_example_per_thread=400,  # Set this to a lower number since a fail translation will cause the whole thread to restart, loosing all the progress of the thread
            large_chunks_threshold=3000,
        )

    # Read function must assign data that has been read to self.data_read
    def read(self) -> None:
        # The read function must call the read function in DataParser class
        # I just want to be sure that the file path is correct
        super(MagpieUltraV01Parser, self).read()

        self.data_read = load_dataset(
            "STEM-AI-mtl/Electrical-engineering", streaming=True
        )

        return None

    # Convert function must assign data that has been converted to self.converted_data
    def convert(self) -> None:
        # The convert function must call the convert function in DataParser class
        # I just want to be sure the read function has actually assigned the self.data_read
        super(MagpieUltraV01Parser, self).convert()

        data_converted = []
        for split in self.data_read:
            for data in tqdm(self.data_read[split], desc=f"Converting {split} data"):
                data_dict = {}
                data_dict["qas_id"] = self.id_generator()
                data_dict["question_text"] = (
                    data["instruction"] + " Question: " + data["input"]
                )
                data_dict["orig_answer_texts"] = data["output"]
                data_converted.append(data_dict)

        # Be sure to assign the final data list to self.converted_data
        self.converted_data = data_converted

        return None


if __name__ == "__main__":
    magpie_ultra_v01_parser = MagpieUltraV01Parser(
        file_path="./samples/dummy.txt",
        output_path="./samples/out",
    )
    magpie_ultra_v01_parser.read()
    magpie_ultra_v01_parser.convert()
    magpie_ultra_v01_parser.save
