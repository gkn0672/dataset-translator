import os


def count_local_records_json(file_path, recursive=True, verbose=False):
    """
    Count records in local JSON files efficiently.

    Args:
        file_path: Directory containing JSON files
        recursive: Whether to search subdirectories
        verbose: Whether to print progress information

    Returns:
        int: Total number of records
    """

    total_count = 0
    json_files = []

    # Find all JSON files
    if recursive:
        for root, _, files in os.walk(file_path):
            for file in files:
                if file.endswith(".json") or file.endswith(".jsonl"):
                    json_files.append(os.path.join(root, file))
    else:
        json_files = [
            os.path.join(file_path, f)
            for f in os.listdir(file_path)
            if f.endswith(".json") or f.endswith(".jsonl")
        ]

    if verbose:
        print(f"Found {len(json_files)} JSON files")

    # Process each file
    for filepath in json_files:
        try:
            # For JSONL format (each line is a JSON object)
            with open(filepath, "r", encoding="utf-8") as f:
                # Just count lines - much faster than parsing JSON
                line_count = sum(1 for line in f if line.strip())
                total_count += line_count

        except Exception as e:
            if verbose:
                print(f"Error processing {filepath}: {e}")

    if verbose:
        print(f"Total records found: {total_count}")

    return total_count
