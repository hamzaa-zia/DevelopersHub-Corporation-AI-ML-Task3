"""Shared helper functions.

Small helper functions keep repeated code out of the main scripts and make the
workflow easier to read.
"""

from pathlib import Path


def create_folder(folder_path: Path) -> None:
    """Create a folder if it does not already exist."""
    folder_path.mkdir(parents=True, exist_ok=True)


def print_section(title: str) -> None:
    """Print a clear title in the terminal."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

