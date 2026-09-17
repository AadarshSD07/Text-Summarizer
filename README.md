# Text Summarizer CLI

A Python CLI tool that reads `.txt` files and generates structured
JSON summaries using a local LLM via Ollama.

Built as a hands-on learning project to understand LLM application
development patterns — including prompt engineering, structured output
parsing, token management, and production error handling.

## What this demonstrates

**Software engineering:**
- Modular design: 8 single-responsibility modules
- Type-safe dataclasses for structured return values
- Generator pattern for memory-efficient file processing

**LLM engineering:**
- System vs user prompt separation
- Few-shot prompting for consistent output format
- Pre-flight token budget checking (tiktoken)
- Retry logic with correction prompts on malformed JSON
- Exponential backoff on API failures

**Production practices:**
- Rotating file logging with severity levels
- Structured error types (ParseError, LLMError, FileError)
- pytest test suite: 67 tests, unit + integration coverage

## Tech stack

Python 3.11 · OpenAI SDK · Ollama · tiktoken · pytest

## What I would do differently at scale

- Replace file-based input with a proper queue (Redis, SQS)
- Add async processing for parallel file handling
- Use Pydantic for schema validation instead of manual checks
- Add a vector store for semantic search across summaries
- Deploy as a FastAPI service with a job queue

## Learning context

This project was built while learning LLM application development.

## Features

- Reads all `.txt` files from a folder automatically
- Structured JSON output: title, 3 bullets, sentiment
- Runs fully locally using Ollama — no API costs, no internet required
- Token counting and context window safety checks before every call
- Intelligent retry with correction prompts on malformed output
- Rotating log files for full diagnostic history
- Clean separation of concerns: one module, one job

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- A pulled model: `ollama pull mistral`

## Setup

```powershell
# Clone and enter the project
cd text-summarizer

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e ".[dev]"

# Configure environment
Copy-Item .env.example .env
# Edit .env if needed (defaults work with Ollama + mistral)
```

## Usage

```powershell
# Summarise all .txt files in texts/ folder
summarize

# Verbose output with token counts and retry details
summarize --verbose

# Custom input and output folders
summarize --input my_articles\ --output results\

# Test LLM connection before processing
summarize --test-connection

# Dry run — scan files without calling the LLM
summarize --dry-run

# Use a specific model
summarize --model phi3

# Control retry attempts
summarize --max-retries 5
```

## Output format

Each `.txt` file produces a `.json` file in the output folder:

```json
{
  "source_file": "article1.txt",
  "title": "James Webb Telescope Reveals Early Universe Galaxies",
  "bullets": [
    "The telescope detected galaxies forming 300 million years after the Big Bang.",
    "Infrared sensors allow observation through cosmic dust clouds.",
    "Carbon dioxide detected in an exoplanet atmosphere may indicate habitable conditions."
  ],
  "sentiment": "positive"
}
```

## Project structure

text-summarizer/
├── summarizer/
│ ├── cli.py # Entry point and argument parsing
│ ├── file_reader.py # Discovers and reads .txt files
│ ├── llm_client.py # LLM API communication
│ ├── prompt_builder.py # System and user prompt construction
│ ├── json_parser.py # Response parsing and validation
│ ├── token_counter.py # Token counting and budget checks
│ ├── retrier.py # Retry logic with correction prompts
│ └── logger_config.py # Logging configuration
├── tests/ # Pytest test suite
├── texts/ # Input .txt files
├── output/ # Generated .json files
├── logs/ # Rotating log files
├── .env.example # Environment template
└── pyproject.toml # Project configuration

## Running tests

```powershell
pytest                                    # All tests
pytest --cov=summarizer                   # With coverage
pytest tests\test_json_parser.py -v      # One module
```