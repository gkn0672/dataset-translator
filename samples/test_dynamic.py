import sys

sys.path.insert(0, r"./")
from config.qa import QAConfig
from translator.parser.dynamic import DynamicDataParser  # Import your modified class
from translator.callback import VerboseCallback
from engine.ollama import OllamaEngine


if __name__ == "__main__":
    magpie_parser = DynamicDataParser(
        file_path=r"samples/out/",
        output_path="./samples/out",
        # dataset_name="argilla/magpie-ultra-v0.1",
        dataset_name=None,
        field_mappings={
            "question": "question",
            "answer": "answer",
            "intention": "intention",  # Fixed typo from "intentiom" to "intention"
        },
        target_config=QAConfig,
        do_translate=True,
        translator=OllamaEngine(model_name="llama3.1:8b-instruct-q4_0"),
        verbose=True,
        parser_callbacks=[VerboseCallback],
        large_chunks_threshold=3000,
        limit=None,
        auto_batch_size=False,  # Enable automatic batch size determination
        max_memory_percent=0.6,  # Use up to 20% of available RAM
        min_batch_size=1,  # Never go below 10 items per batch
        max_batch_size=5,  # Never go above 1000 items per batch
    )

    magpie_parser.read()
    magpie_parser.convert()
    magpie_parser.save()
