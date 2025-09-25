"""Sample Firecrawl extraction payloads used when API access is unavailable."""

from importlib import resources
from pathlib import Path


def data_path(filename: str) -> Path:
    """Return the on-disk path for a sample data asset."""

    with resources.as_file(resources.files(__package__) / filename) as path:
        return path

