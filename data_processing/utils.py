import json
import logging
from typing import Dict, Any


# Helper functions
def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and return content from a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"{file_path} not found.")
        raise RuntimeError(f"{file_path} file not found.")
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from {file_path}")
        raise RuntimeError(f"Error decoding JSON from {file_path}")


def load_text_file(file_path: str) -> str:
    """Load and return text content from a file."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"{file_path} not found.")
        raise RuntimeError(f"{file_path} file not found.")
