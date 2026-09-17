"""
logger_config.py — Centralised logging configuration for text-summarizer.

Call setup_logging() once at application startup (in cli.py main()).
After that, every module gets its logger with:

    import logging
    logger = logging.getLogger(__name__)

And it just works — logs go to both the console and a rotating file.

Design decisions:
  - Console handler: WARNING+ only (clean user experience)
  - File handler:    DEBUG+ always (full diagnostic history)
  - Rotating file:   max 1MB per file, keep 3 backups (disk-safe)
  - Format differs:  console is minimal, file has full context
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path


# The root logger name for the entire package.
# All summarizer.* loggers are children of this.
PACKAGE_LOGGER_NAME = "summarizer"


def setup_logging(
    log_dir: Path = Path("logs"),
    console_level: int = logging.WARNING,
    file_level: int = logging.DEBUG,
    verbose: bool = False,
) -> logging.Logger:
    """
    Configure and return the root package logger.

    Sets up two handlers:
      1. StreamHandler  → stderr, WARNING+ (or DEBUG if verbose=True)
      2. RotatingFileHandler → logs/summarizer_YYYY-MM-DD.log, DEBUG+

    Args:
        log_dir:       Directory for log files (created if absent)
        console_level: Minimum level for console output
        file_level:    Minimum level for file output
        verbose:       If True, drops console level to DEBUG

    Returns:
        The configured root package logger
    """
    # Create log directory — won't fail if it already exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get the root package logger
    # All child loggers (summarizer.cli, summarizer.llm_client, etc.)
    # inherit this configuration automatically
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    # Set the logger's own level to the most permissive of the two handlers
    # (the handlers apply their own filters — logger must let everything through)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if setup_logging() is called more than once
    if logger.handlers:
        logger.handlers.clear()

    # ── Handler 1: Console (stderr) ───────────────────────────────────
    # Goes to stderr, not stdout, so it doesn't pollute piped output
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG if verbose else console_level)

    # Console format: minimal — level and message only
    # We don't need timestamps or module names in the terminal
    console_formatter = logging.Formatter(
        fmt="%(levelname)-8s %(message)s",
        # Example output:
        # WARNING  Article truncated: article3.txt exceeded token limit
        # ERROR    Parse failed after 3 retries: article3.txt
    )
    console_handler.setFormatter(console_formatter)

    # ── Handler 2: Rotating file ──────────────────────────────────────
    # Filename includes today's date for easy searching
    log_filename = log_dir / f"summarizer_{datetime.now():%Y-%m-%d}.log"

    # RotatingFileHandler: caps file size at maxBytes, keeps backupCount
    # old files. When the current file hits 1MB, it renames it to
    # summarizer.log.1, summarizer.log.2 etc. and starts fresh.
    # This prevents log files from growing unbounded on long runs.
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_filename,
        maxBytes=1_000_000,     # 1 MB per file
        backupCount=3,          # Keep 3 old files (4 files total max)
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)

    # File format: full context — timestamp, level, module, line, message
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # Example output:
        # 2025-08-01 14:23:07 | DEBUG    | summarizer.llm_client:87 | Sending 847 tokens to mistral
        # 2025-08-01 14:23:09 | INFO     | summarizer.cli:142 | Saved output/article1.json
        # 2025-08-01 14:23:11 | WARNING  | summarizer.retrier:88 | Retry 2/3 for article2.txt
        # 2025-08-01 14:23:13 | ERROR    | summarizer.retrier:134 | All retries failed: article3.txt
    )
    file_handler.setFormatter(file_formatter)

    # Attach both handlers to the root package logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function — get a named child logger.

    Usage in any module:
        from summarizer.logger_config import get_logger
        logger = get_logger(__name__)

    This is equivalent to logging.getLogger(__name__) but
    keeps the import explicit and project-specific.

    Args:
        name: Usually __name__ — the calling module's full path

    Returns:
        A child logger that inherits from the package root logger
    """
    return logging.getLogger(name)