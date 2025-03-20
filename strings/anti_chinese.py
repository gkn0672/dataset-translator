import re


def clean_chinese_mix(text):
    """
    Smart text cleaning that:
    1. Selectively removes Chinese characters only when they're mixed within words
    2. Preserves standalone Chinese words/phrases
    3. Fixes technical terms
    4. Normalizes spacing

    Args:
        text (str): Raw translated text

    Returns:
        str: Cleaned text with appropriate handling of Chinese characters
    """
    if not text:
        return text

    # Step 1: Selectively handle Chinese characters
    def selective_chinese_handling(text):
        # Helper to identify Chinese characters
        def is_chinese_char(char):
            cp = ord(char)
            # Main CJK Unified Ideographs block
            if 0x4E00 <= cp <= 0x9FFF:
                return True
            return False

        # Helper to determine if a character is a letter (Latin or Vietnamese)
        def is_letter(char):
            # Latin letters and Vietnamese characters
            return re.match(r"[a-zA-ZÀ-ỹ]", char) is not None

        # Process text word by word
        words = re.findall(r"\S+", text)
        result = []

        for word in words:
            # If the word contains both Chinese and Latin/Vietnamese characters, remove only the Chinese
            has_chinese = any(is_chinese_char(c) for c in word)
            has_letters = any(is_letter(c) for c in word)

            if has_chinese and has_letters:
                # Mixed word: remove Chinese characters
                cleaned_word = "".join(c for c in word if not is_chinese_char(c))
                result.append(cleaned_word)
            else:
                # Pure word (either all Chinese or no Chinese): keep as is
                result.append(word)

        # Rejoin with spaces
        return " ".join(result)

    # Normalize spacing
    def normalize_spacing(text):
        # Replace multiple spaces with a single space
        result = re.sub(r" +", " ", text)

        # Ensure no spaces before punctuation
        result = re.sub(r" ([,.!?:;)])", r"\1", result)

        # Ensure space after punctuation if followed by a letter
        result = re.sub(r"([,.!?:;(])([a-zA-ZÀ-ỹ])", r"\1 \2", result)

        # Fix spaces around math expressions
        result = re.sub(r"(\$[^$]*\$)", lambda m: m.group(1).replace(" ", ""), result)

        return result.strip()

    # Apply the pipeline
    cleaned = selective_chinese_handling(text)
    cleaned = normalize_spacing(cleaned)

    return cleaned
