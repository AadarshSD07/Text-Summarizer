"""
text-summarizer
===============
A CLI tool that reads .txt files and summarizes them using a local LLM.

Package structure:
    cli.py           — Entry point, argument parsing
    file_reader.py   — Discovers and reads .txt files
    llm_client.py    — Communicates with the LLM API
    prompt_builder.py — Constructs system and user prompts
    json_parser.py   — Parses and validates structured JSON output
    token_counter.py — Estimates token usage before API calls
"""

# This file's existence is what makes Python treat this folder
# as a package (importable module) rather than a plain directory.
#
# When Python sees: from summarizer import file_reader
# It looks for summarizer/__init__.py first, then summarizer/file_reader.py
#
# Leaving it mostly empty is fine and conventional.
# Some teams put shared constants or version info here.

__version__ = "0.1.0"
__author__ = "Aadarsh"