import inspect
from typing import List, Dict


def get_mandatory_fields(config_class) -> List[str]:
    """
    Identify mandatory fields from a dataclass-based configuration.
    """
    # Get field definitions from the dataclass
    mandatory_fields = []
    for name, param in inspect.signature(config_class.__init__).parameters.items():
        # Skip self parameter
        if name == "self":
            continue

        # If the parameter has no default value (no default=) and no default_factory
        # then it's a mandatory field
        if param.default is param.empty:
            mandatory_fields.append(name)

    return mandatory_fields


def validate_mappings(
    field_mappings: Dict[str, str], mandatory_fields: List[str]
) -> None:
    """
    Validate that all mandatory fields have mappings.
    """
    missing_fields = []
    for field in mandatory_fields:
        # Skip qas_id as it's handled specially if not mapped
        if field != "qas_id" and field not in field_mappings:
            missing_fields.append(field)

    if missing_fields:
        raise ValueError(
            f"The following mandatory fields are missing from field_mappings: {', '.join(missing_fields)}. "
            f"All mandatory fields ({', '.join(mandatory_fields)}) must be mapped."
        )


def patch_target_config(
    target_config, additional_fields, original_get_keys=None, original_annotations=None
):
    """
    Temporarily patch the target_config class to include additional fields.
    This handles both the get_keys method and type annotations.
    """
    # Save original methods and annotations if not provided
    if original_get_keys is None:
        original_get_keys = target_config.get_keys
    if original_annotations is None:
        original_annotations = getattr(target_config, "__annotations__", {}).copy()

    # Create extended get_keys method
    extended_keys = original_get_keys() + additional_fields

    def extended_get_keys(cls):
        return extended_keys

    # Patch the get_keys method
    target_config.get_keys = classmethod(extended_get_keys)

    # Patch the annotations
    # Add string type annotations for additional fields
    updated_annotations = original_annotations.copy()
    for field in additional_fields:
        updated_annotations[field] = str

    # Update the class annotations
    target_config.__annotations__ = updated_annotations

    return original_get_keys, original_annotations


def restore_target_config(target_config, original_get_keys, original_annotations):
    """
    Restore the original target_config class attributes.
    """
    # Restore original get_keys method
    target_config.get_keys = original_get_keys

    # Restore original annotations
    target_config.__annotations__ = original_annotations
