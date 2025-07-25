import sys

sys.path.insert(0, r"./")
from tqdm.auto import tqdm

from datasets import load_dataset

from config.qa import QAConfig
from translator.parser.base import BaseParser
from translator.callback import VerboseCallback
from engine.ollama import OllamaEngine


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
            no_translated_code=False,
            translator=OllamaEngine(model_name="llama3.1:8b-instruct-q4_0"),
            parser_callbacks=[VerboseCallback],
            max_example_per_thread=400,
            large_chunks_threshold=3000,
        )

    # Read function must assign data that has been read to self.data_read
    def read(self) -> None:
        super(MagpieUltraV01Parser, self).read()

        self.data_read = load_dataset(
            "STEM-AI-mtl/Electrical-engineering", streaming=True
        )
        return None

    # Convert function must assign data that has been converted to self.converted_data
    def convert(self) -> None:
        super(MagpieUltraV01Parser, self).convert()

        data_converted = []
        for split in self.data_read:
            for data in tqdm(self.data_read[split], desc=f"Converting {split} data"):
                data_dict = {}
                data_dict["qas_id"] = self.id_generator()
                data_dict["question"] = (
                    data["instruction"] + " Question: " + data["input"]
                )
                data_dict["answer"] = data["output"]
                data_converted.append(data_dict)

        # Be sure to assign the final data list to self.converted_data
        self.converted_data = data_converted[:10]

        return None


if __name__ == "__main__":
    magpie_ultra_v01_parser = MagpieUltraV01Parser(
        file_path="./samples/dummy.txt",
        output_path="./samples/out",
    )
    magpie_ultra_v01_parser.read()
    magpie_ultra_v01_parser.convert()
    magpie_ultra_v01_parser.save
