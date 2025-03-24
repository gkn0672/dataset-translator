import sys

sys.path.insert(0, r"./")
from config.qa import QAConfig
from translator.parser.dynamic import DynamicDataParser
from translator.callback import VerboseCallback
from engine.ollama import OllamaEngine


if __name__ == "__main__":
    magpie_parser = DynamicDataParser(
        file_path="./samples/dummy.txt",
        output_path="./samples/out",
        dataset_name="argilla/magpie-ultra-v0.1",
        field_mappings={
            "question": "instruction",
            "answer": "response",
        },
        target_config=QAConfig,
        do_translate=True,
        translator=OllamaEngine(model_name="llama3.1:8b-instruct-q4_0"),
        verbose=False,
        parser_callbacks=[VerboseCallback],
        max_example_per_thread=25,
        large_chunks_threshold=3000,
        limit=10,
    )

    magpie_parser.read()
    magpie_parser.convert()
    magpie_parser.save
