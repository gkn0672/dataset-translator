from typing import Dict, List, Union
import threading

print_lock = threading.Lock()


def translate_example(
    example: Dict,
    fields_to_translate: List[str],
    translator,
    source_lang: str,
    target_lang: str,
    max_example_length: int,
    fail_translation_code: str,
    verbose: bool = False,
    enable_sub_task_thread: bool = False,
    max_list_length_per_thread: int = 100,
    cancellation_event=None,
) -> Dict:
    """
    Translate a single example.

    Args:
        example: The example to translate
        fields_to_translate: List of field names to translate
        translator: Translator instance
        source_lang: Source language code
        target_lang: Target language code
        max_example_length: Maximum length of text to translate
        fail_translation_code: Code to return on translation failure
        verbose: Whether to print verbose logs
        enable_sub_task_thread: Whether to use threading for large lists
        max_list_length_per_thread: Maximum list length to process in a single thread
        cancellation_event: Optional event to check for cancellation

    Returns:
        Translated example
    """
    # Check for cancellation
    if cancellation_event and cancellation_event.is_set():
        return example

    result = example.copy()

    # Translate each field that should be translated
    for field in fields_to_translate:
        # Check for cancellation
        if cancellation_event and cancellation_event.is_set():
            return result

        if field not in example or example[field] is None or example[field] == "":
            continue

        # Translate the field
        value = example[field]
        if isinstance(value, list):
            # Handle list values
            for i, item in enumerate(value):
                if len(item) > max_example_length:
                    if verbose:
                        print(
                            f"Truncating a list item in field {field} as it exceeds max length"
                        )
                    value[i] = item[:max_example_length]

            # Check if we need special handling for large lists
            if enable_sub_task_thread and len(value) >= max_list_length_per_thread:
                result[field] = translate_large_list(
                    value,
                    translator,
                    source_lang,
                    target_lang,
                    fail_translation_code,
                    max_list_length_per_thread,
                    cancellation_event,
                )
            else:
                result[field] = translator.translate(
                    value,
                    src=source_lang,
                    dest=target_lang,
                    fail_translation_code=fail_translation_code,
                )
        else:
            # Handle string values
            if len(value) > max_example_length:
                if verbose:
                    print(f"Truncating field {field} as it exceeds max length")
                value = value[:max_example_length]

            result[field] = translator.translate(
                value,
                src=source_lang,
                dest=target_lang,
                fail_translation_code=fail_translation_code,
            )

    return result


def translate_large_list(
    items: List[str],
    translator,
    source_lang: str,
    target_lang: str,
    fail_translation_code: str,
    max_list_length_per_thread: int,
    cancellation_event=None,
) -> List[str]:
    """
    Handle translation of a large list by breaking it into smaller chunks.
    """
    # Split the list into chunks
    chunks = split_list(items, max_list_length_per_thread)
    results = []

    for chunk in chunks:
        # Check for cancellation
        if cancellation_event and cancellation_event.is_set():
            return items  # Return original items if cancelled

        translated = translator.translate(
            chunk,
            src=source_lang,
            dest=target_lang,
            fail_translation_code=fail_translation_code,
        )
        results.extend(translated)

    return results


def split_list(items: List, chunk_size: int) -> List[List]:
    """
    Split a list into chunks of a given size.
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def translate_batch(
    data_batch: List[Dict],
    fields_to_translate: List[str],
    translator,
    source_lang: str,
    target_lang: str,
    max_example_length: int,
    fail_translation_code: str,
    verbose: bool = False,
    enable_sub_task_thread: bool = False,
    max_list_length_per_thread: int = 100,
    desc: str = None,
    cancellation_event=None,
) -> Union[None, List[Dict]]:
    """
    Translate a batch of data.
    """
    if not data_batch:
        return []

    # Check for cancellation at the beginning
    if cancellation_event and cancellation_event.is_set():
        return []

    result = []
    # Fix the duplication of "Batch" in the output
    batch_desc = f"Translating {desc}" if desc else "Translating Batch"

    # Use a lock to prevent interleaved output from multiple batches
    with print_lock:
        separator = "=" * 80
        print(f"\n{separator}")
        print(f"  STARTING {batch_desc.upper()} - {len(data_batch)} ITEMS")
        print(f"{separator}\n")

    # Process each example
    for idx, example in enumerate(data_batch):
        # Check for cancellation
        if cancellation_event and cancellation_event.is_set():
            print(f"❌ Cancellation detected during batch translation: {batch_desc}")
            return []

        # Translate the example
        translated_example = translate_example(
            example,
            fields_to_translate,
            translator,
            source_lang,
            target_lang,
            max_example_length,
            fail_translation_code,
            verbose,
            enable_sub_task_thread,
            max_list_length_per_thread,
            cancellation_event,
        )
        result.append(translated_example)

        # Print progress with lock to avoid interleaving
        current = idx + 1
        total = len(data_batch)
        if current == 1 or current == total or current % (5 if total > 10 else 1) == 0:
            with print_lock:
                percent = int(current / total * 100)
                print(f"{batch_desc}: Item {current}/{total} ({percent}%) completed")

    # Check for cancellation before completion
    if cancellation_event and cancellation_event.is_set():
        return result

    # Print completion message with lock to avoid interleaving
    with print_lock:
        print(f"\n{separator}")
        print(
            f"  COMPLETED {batch_desc.upper()} - {len(data_batch)}/{len(data_batch)} ITEMS"
        )
        print(f"{separator}\n")

    return result
