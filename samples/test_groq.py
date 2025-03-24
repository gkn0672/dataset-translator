# TODO: experiment with data read with stream mode
import sys

sys.path.insert(0, r"./")
from tqdm.auto import tqdm

from datasets import load_dataset

from config.qa import QAConfig
from translator.parser.base import BaseParser
from translator.callback import VerboseCallback
from engine.groq import GroqEngine


PARSER_NAME = "MagpieUltraV01"


# Patience is the key since the data is large and is using an LLM based translator
class MagpieUltraV01Parser(BaseParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(
            file_path,
            output_path,
            parser_name=PARSER_NAME,
            target_config=QAConfig,
            target_fields=[
                "question",
                "answer",
            ],
            do_translate=True,
            no_translated_code=False,  # Remove any instance of string that appears to be coding language (e.g. Python code, HTML, etc.)
            translator=GroqEngine(),  # Groq is very slow but it is a high quality translator
            parser_callbacks=[VerboseCallback],
            max_example_per_thread=25,
            large_chunks_threshold=3000,
        )

    # Read function must assign data that has been read to self.data_read
    def read(self) -> None:
        super(MagpieUltraV01Parser, self).read()

        self.data_read = load_dataset("argilla/magpie-ultra-v0.1", streaming=True)

        return None

    # Convert function must assign data that has been converted to self.converted_data
    def convert(self) -> None:
        super(MagpieUltraV01Parser, self).convert()

        data_converted = []
        for split in self.data_read:
            for data in tqdm(self.data_read[split], desc=f"Converting {split} data"):
                data_dict = {}
                data_dict["qas_id"] = self.id_generator()
                data_dict["question"] = data["instruction"]
                data_dict["answer"] = data["response"]
                data_converted.append(data_dict)

        # Be sure to assign the final data list to self.converted_data
        self.converted_data = data_converted[:100]  # 100 examples for testing purposes

        return None


if __name__ == "__main__":
    magpie_ultra_v01_parser = MagpieUltraV01Parser(
        file_path="./samples/dummy.txt",
        output_path="./samples/out",
    )
    magpie_ultra_v01_parser.read()
    magpie_ultra_v01_parser.convert()
    magpie_ultra_v01_parser.save
