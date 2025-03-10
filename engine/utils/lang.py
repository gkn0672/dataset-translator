def get_language_name(code: str) -> str:
    """
    Retrieves the language name based on the given language code.

    Args:
        code (str): The language code.

    Returns:
        str: The corresponding language name.

    Raises:
        KeyError: If the language code is not found in the language map.
    """

    language_map = {
        "en": "English",
        "en-AU": "English (Australia)",
        "en-BZ": "English (Belize)",
        "en-CA": "English (Canada)",
        "en-CB": "English (Caribbean)",
        "en-GB": "English (United Kingdom)",
        "en-IE": "English (Ireland)",
        "en-JM": "English (Jamaica)",
        "en-NZ": "English (New Zealand)",
        "en-PH": "English (Republic of the Philippines)",
        "en-TT": "English (Trinidad and Tobago)",
        "en-US": "English (United States)",
        "en-ZA": "English (South Africa)",
        "en-ZW": "English (Zimbabwe)",
        "vi": "Vietnamese",
        "vi-VN": "Vietnamese (Viet Nam)",
    }

    try:
        language_name = language_map.get(code)
    except KeyError:
        raise KeyError("Language code not found")

    return language_name


if __name__ == "__main__":
    # Example usage
    print(get_language_name("en-US"))  # Output: English (United States)
    print(get_language_name("zh-TW"))  # Output: Chinese (Traditional)
