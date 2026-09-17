"""
file_reader.py - Discovers and safely reads .txt files from a folder.

Single responsibility: given a folder path, find all .txt files
and return their contents in a structured, type-safe format.

This module knows NOTHING about LLMs, JSON, or the CLI.
It only knows how to read files. That's the whole job.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator
from summarizer.logger_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileContent:
    """
    Represents one successfully-read text file.

    Using a dataclass instead of a plain dict or tuple because:
    - Named fields are self-documenting (result.name vs result[0])
    - Type hints catch bugs at development time
    - IDE autocomplete works properly
    - Easy to add fields later without breaking callers
    """
    name: str          # Filename without extension: "article1"
    filename: str      # Full filename with extension: "article1.txt"
    content: str       # The actual text content
    path: Path         # Full path to the file
    char_count: int    # Length of content in characters
    encoding_used: str # Which encoding successfully read the file


@dataclass
class FileError:
    """
    Represents a file that could NOT be read.

    Why return an error object instead of raising an exception?
    Because we want to process ALL files and report ALL errors
    at the end, not crash on the first problem.

    This pattern is called "error accumulation" — collect errors,
    continue processing, report everything at the end.
    """
    filename: str      # Which file failed
    path: Path         # Where it was
    error_type: str    # "permission_denied", "encoding_error", etc.
    message: str       # Human-readable explanation


# A type alias: ReadResult is either a good file or an error
# This makes function signatures much more readable
ReadResult = FileContent | FileError


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def discover_txt_files(folder: Path) -> list[Path]:
    """
    Find all .txt files in a folder, sorted alphabetically.

    Why sorted? Because the order files are processed in affects
    the order of output JSON. Alphabetical gives deterministic,
    reproducible results — important for testing and debugging.

    Args:
        folder: Path to the folder to search (non-recursive)

    Returns:
        List of Path objects, one per .txt file, sorted by name.

    Raises:
        NotADirectoryError: If folder exists but is a file, not a folder.
        FileNotFoundError: If folder does not exist at all.
    """
    # We validate existence in cli.py, but defensive check here too.
    # Two layers of validation = two chances to catch problems early.
    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: '{folder}'\n"
            f"Current working directory: {Path.cwd()}"
        )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"'{folder}' is a file, not a folder."
        )

    # Path.glob("*.txt") returns a generator of Path objects
    # matching the pattern in this folder (non-recursive).
    # sorted() forces alphabetical order.
    txt_files = sorted(folder.glob("*.txt"))
    logger.debug(f"Found {len(txt_files)} .txt file(s) in '{folder}'")
    return txt_files


def read_single_file(file_path: Path) -> ReadResult:
    """
    Read one .txt file as safely as possible.

    Encoding strategy (this is the core of defensive file reading):
      1. Try UTF-8 first — it's the modern standard, handles
         all Unicode characters including emoji, accents, etc.
      2. If UTF-8 fails (UnicodeDecodeError), try latin-1 (iso-8859-1).
         latin-1 maps every possible byte value to a character,
         so it NEVER raises UnicodeDecodeError. It's a guaranteed
         fallback, though some characters may look wrong.
      3. If even file access fails (permission error, locked file,
         disk error), return a FileError instead of crashing.

    Args:
        file_path: Path to the specific .txt file to read

    Returns:
        FileContent if successful, FileError if not.
        NEVER raises an exception — all errors become FileError objects.
    """
    # --- Attempt 1: UTF-8 (preferred) ---
    try:
        content = file_path.read_text(encoding="utf-8")
        return FileContent(
            name=file_path.stem,          # "article1" (no extension)
            filename=file_path.name,      # "article1.txt"
            content=content.strip(),      # Remove leading/trailing whitespace
            path=file_path,
            char_count=len(content.strip()),
            encoding_used="utf-8",
        )

    except UnicodeDecodeError:
        # File has bytes that aren't valid UTF-8.
        # Common causes: file was saved on an old Windows system
        # (Windows-1252), or contains special characters from
        # another encoding. Try latin-1 as a fallback.
        logger.debug(f"UTF-8 failed for '{file_path.name}', trying latin-1")
        pass  # Intentional: fall through to attempt 2

    # --- Attempt 2: latin-1 fallback ---
    try:
        content = file_path.read_text(encoding="latin-1")
        logger.warning(
            f"'{file_path.name}' required latin-1 fallback encoding. "
            f"File may have been saved on an older Windows system."
        )
        return FileContent(
            name=file_path.stem,
            filename=file_path.name,
            content=content.strip(),
            path=file_path,
            char_count=len(content.strip()),
            encoding_used="latin-1",  # Caller knows it was a fallback
        )

    except PermissionError:
        # OS denied access to the file.
        # Common on Windows: file is open in another program,
        # or the user doesn't have read permissions.
        logger.error(f"Permission denied reading '{file_path.name}'")
        return FileError(
            filename=file_path.name,
            path=file_path,
            error_type="permission_denied",
            message=(
                f"Cannot read '{file_path.name}': permission denied.\n"
                f"Check that the file isn't open in another program."
            ),
        )

    except OSError as e:
        # Catch-all for other OS-level errors: disk errors,
        # network drives disconnecting, etc.
        logger.error(f"OS error reading '{file_path.name}': {e}")
        return FileError(
            filename=file_path.name,
            path=file_path,
            error_type="os_error",
            message=f"Cannot read '{file_path.name}': {e}",
        )


def read_txt_files(
    folder: Path,
    max_files: int | None = None,
) -> Generator[ReadResult, None, None]:
    """
    Generator that yields FileContent or FileError for each .txt file found.

    Why a generator instead of returning a list?
    - Memory efficient: only one file in RAM at a time
    - Caller can process files as they arrive (streaming)
    - Caller can break early without reading all files
    - For this project: habit-forming — large-scale LLM pipelines
      use streaming patterns everywhere

    Args:
        folder:    Path to the folder containing .txt files
        max_files: If set, stop after this many files (for --max-files CLI flag)

    Yields:
        FileContent for each successfully-read file
        FileError for each file that could not be read

    Example usage:
        for result in read_txt_files(Path("texts")):
            if isinstance(result, FileContent):
                print(f"Read {result.filename}: {result.char_count} chars")
            elif isinstance(result, FileError):
                print(f"Failed: {result.message}")
    """
    txt_files = discover_txt_files(folder)

    # Apply max_files limit if specified
    if max_files is not None:
        txt_files = txt_files[:max_files]

    if not txt_files:
        # Not an error — just nothing to do.
        # The CLI layer will warn the user separately.
        return

    for file_path in txt_files:
        # read_single_file never raises — it always returns ReadResult
        result = read_single_file(file_path)
        yield result   # Hand this result to the caller, then pause


def summarize_read_results(
    results: list[ReadResult],
) -> dict:
    """
    Given a list of results, produce a summary dict for reporting.

    Useful for the --verbose flag and for end-of-run reporting.

    Args:
        results: List of FileContent and FileError objects

    Returns:
        Dict with counts and lists of successes/failures
    """
    successes = [r for r in results if isinstance(r, FileContent)]
    failures  = [r for r in results if isinstance(r, FileError)]

    return {
        "total":     len(results),
        "succeeded": len(successes),
        "failed":    len(failures),
        "successes": successes,
        "failures":  failures,
        "total_chars": sum(r.char_count for r in successes),
    }