"""Checklist definitions and validation for the hybrid verifier."""

CATEGORIES = {"required", "prohibited", "incidental"}

CHECK_TYPES = {
    "file_exists",
    "file_absent",
    "file_changed",
    "file_unchanged",
    "semantic",
} 

def validate_item(item):
    """Reject checklist items with missing or invalid fields."""
    if not isinstance(item, dict):
        raise ValueError("Checklist item must be a dictionary.")

    expected_fields = {
        "id", "category", "description", "check_type", "parameters"
    }

    if set(item) != expected_fields:
        raise ValueError(
            f"Checklist item must contain exactly: {sorted(expected_fields)}"
        )

    for field in ["id", "category", "description", "check_type"]:
        if not isinstance(item[field], str) or not item[field].strip():
            raise ValueError(f"{field} must be a nonempty string.")

    if item["category"] not in CATEGORIES:
        raise ValueError("Invalid checklist category.")

    if item["check_type"] not in CHECK_TYPES:
        raise ValueError("Invalid check type.")

    if not isinstance(item["parameters"], dict):
        raise ValueError("parameters must be a dictionary.")

    parameters = item["parameters"]

    if item["check_type"] == "semantic":
        # The description specifies the condition to evaluate.
        if parameters:
            raise ValueError("semantic checks must have empty parameters.")
    else:
        # File checks need a path identifying the file to inspect.
        if set(parameters) != {"expected_path"}:
            raise ValueError(
                "File checks must have exactly one parameter: expected_path."
            )

        path = parameters["expected_path"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError("expected_path must be a nonempty string.")