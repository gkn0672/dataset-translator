import sys

sys.path.insert(0, r"./")
from config.qa import QAConfig
from translator.parser.dynamic import DynamicDataParser  # Import your modified class
from translator.callback.verbose import VerboseCallback
from translator.callback.huggingface import HuggingFaceCallback
from engine.ollama import OllamaEngine


if __name__ == "__main__":
    magpie_parser = DynamicDataParser(
        file_path=None,
        output_path="./samples/out",
        dataset_name="argilla/magpie-ultra-v0.1",
        # dataset_name=None,
        field_mappings={
            "question": "instruction",
            "answer": "response",
            "intention": "intent",  # Fixed typo from "intentiom" to "intention"
        },
        target_config=QAConfig,
        do_translate=True,
        translator=OllamaEngine(model_name="llama3.1:8b-instruct-q4_0"),
        verbose=True,
        parser_callbacks=[VerboseCallback, HuggingFaceCallback],
        large_chunks_threshold=3000,
        limit=10,
        auto_batch_size=False,
        max_memory_percent=0.6,
        min_batch_size=1,
        max_batch_size=5,
    )

    magpie_parser.read()
    magpie_parser.convert()
    magpie_parser.save()
