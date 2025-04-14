import json
from dataclasses import fields
from config.qa import QAConfig
from config.cot import COTConfig


def get_config_fields(config_class):
    """
    Get field names from a config class.

    Args:
        config_class: QAConfig or COTConfig class (not instance)

    Returns:
        List of field names, excluding 'qas_id'
    """
    all_fields = fields(config_class)
    return [f.name for f in all_fields if f.name != "qas_id"]


def generate_config_template(config_type):
    """
    Generate a template JSON mapping with fields from the specified config type.

    Args:
        config_type (str): Type of configuration ("QAConfig" or "COTConfig")

    Returns:
        str: JSON string with template
    """
    if config_type == "QAConfig":
        config_fields = get_config_fields(QAConfig)
    elif config_type == "COTConfig":
        config_fields = get_config_fields(COTConfig)
    else:
        return "{}"

    mapping = {field: "" for field in config_fields}
    return json.dumps(mapping, indent=2)


def update_json_mapping(
    target_config,
    qa_question,
    qa_answer,
    cot_question,
    cot_reasoning,
    cot_answer,
    additional_fields,
):
    """
    Update JSON mapping based on current field values.

    Args:
        target_config (str): Configuration type
        qa_question (str): QA question field value
        qa_answer (str): QA answer field value
        cot_question (str): COT question field value
        cot_reasoning (str): COT reasoning field value
        cot_answer (str): COT answer field value
        additional_fields (list): List of additional field dictionaries

    Returns:
        str: Updated JSON mapping as a string
    """
    mapping = {}

    # Add mandatory fields based on config type
    if target_config == "QAConfig":
        mapping["question"] = qa_question
        mapping["answer"] = qa_answer
    else:  # COTConfig
        mapping["question"] = cot_question
        mapping["reasoning"] = cot_reasoning
        mapping["answer"] = cot_answer

    # Add additional fields
    for field in additional_fields:
        if field["key"] and field["key"] not in mapping:
            mapping[field["key"]] = field["value"]

    return json.dumps(mapping, indent=2)
