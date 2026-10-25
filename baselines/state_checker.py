def check_file_exists(expected_path, observed_files, listing_complete=False):
    """Check whether a file exists using evidence from the agent's environment."""
    if observed_files is None:
        return {
            "status": "unknown",
            "reason": "No file listing is available.",
        }

    if expected_path in observed_files:
        return {
            "status": "pass",
            "reason": f"Found {expected_path} in the file listing.",
        }

    if not listing_complete:
        return {
            "status": "unknown",
            "reason": "The file is absent, but the listing is incomplete.",
        }

    return {
        "status": "fail",
        "reason": f"{expected_path} is absent from the complete listing.",
    }

def check_file_changed(before_hash, after_hash):
    """Compare hashes of the same file, computed using the same algorithm."""
    if before_hash is None or after_hash is None:
        return {
            "status": "unknown",
            "reason": "A before or after hash is unavailable.",
        }

    if before_hash != after_hash:
        return {
            "status": "pass",
            "reason": "The file contents changed.",
        }

    return {
        "status": "fail",
        "reason": "The file contents did not change.",
    }