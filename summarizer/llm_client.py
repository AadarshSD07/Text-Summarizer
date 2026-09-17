"""
llm_client.py - Handles all communication with the LLM API.

Single responsibility: given a prompt (system + user messages),
call the LLM and return the raw text response.

This module knows nothing about files, JSON parsing, or the CLI.
It only knows how to talk to the LLM. That isolation means if you
switch from Ollama to OpenAI to Groq, you change ONE file.

Design decision: we use the OpenAI Python SDK pointed at Ollama's
base_url. This is the "OpenAI-compatible" pattern - Ollama speaks
the same HTTP API as OpenAI, so the same SDK works for both.
You switch providers by changing two environment variables.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from openai import (
    OpenAI,
    APIConnectionError,    # Can't reach the server at all
    APITimeoutError,       # Server took too long to respond
    APIStatusError,        # Server responded with an error code (4xx, 5xx)
    RateLimitError,        # Too many requests (relevant for OpenAI, not Ollama)
)
from summarizer.logger_config import get_logger
from summarizer.prompt_builder import PromptPair

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """
    Represents a successful response from the LLM.

    Wrapping the raw response in a dataclass gives us:
    - Type safety: callers know exactly what fields exist
    - Token tracking: usage data is captured alongside content
    - Testability: easy to create fake LLMResponse objects in tests
    """
    content: str            # The model's actual text response
    model: str              # Which model was used (may differ from requested)
    prompt_tokens: int      # Tokens in your input (system + user messages)
    completion_tokens: int  # Tokens in the model's response
    total_tokens: int       # Sum of above


@dataclass
class LLMError:
    """
    Represents a failed LLM call.

    Same philosophy as FileError in file_reader.py:
    return a structured error instead of crashing the program.
    The caller decides whether to retry, skip, or abort.
    """
    error_type: str    # "connection", "timeout", "rate_limit", "api_error"
    message: str       # Human-readable explanation
    status_code: int | None = None  # HTTP status code if available
    retryable: bool = True          # Should the caller try again?


# Union type: a call returns either success or error
LLMResult = LLMResponse | LLMError


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    """
    Create and return a configured OpenAI client.

    Why a function instead of a module-level global client?
    - load_dotenv() must run BEFORE we read env vars
    - If client were created at import time, dotenv might not be loaded yet
    - A factory function is called on demand, after dotenv is loaded

    The client is configured via environment variables so that
    switching from Ollama to OpenAI requires zero code changes.
    """
    api_key  = os.getenv("OPENAI_API_KEY",  "ollama")
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

    # The OpenAI SDK requires api_key to be non-empty.
    # Ollama doesn't actually validate it, but the SDK checks locally.
    # We default to "ollama" as a harmless placeholder.
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,   # 60 second timeout — local models can be slow
                        # to generate the first token (model loading)
    )

    return client


# ---------------------------------------------------------------------------
# Core call function
# ---------------------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 600,
) -> LLMResult:
    """
    Make one call to the LLM and return the result.

    This is the core function of the entire project. Everything
    else is scaffolding around this one operation.

    Args:
        system_prompt: Instructions that define the model's behaviour.
                       Sent as role="system". Stays constant across files.
        user_prompt:   The actual content to process (the article text).
                       Sent as role="user". Changes per file.
        model:         Model name. If None, reads OPENAI_MODEL from .env.
        temperature:   Randomness dial. 0.0 = deterministic, 1.0 = creative.
                       0.3 is good for structured summarization.
        max_tokens:    Hard cap on response length in tokens.
                       600 tokens ≈ 450 words — plenty for our JSON output.

    Returns:
        LLMResponse on success, LLMError on failure.
        Never raises an exception.
    """
    # Resolve model: CLI arg → .env file → hardcoded default
    # This priority chain lets users override at any level
    resolved_model = (
        model
        or os.getenv("OPENAI_MODEL")
        or "mistral"
    )

    # Build the client fresh each call.
    # For a production system you'd cache this, but for a CLI
    # tool that processes a handful of files, it's fine.
    client = build_client()

    # The messages array: the full conversation context.
    # For summarization, we always have exactly two messages:
    # system (instructions) + user (the article content).
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    # Attempt the API call with specific exception handling.
    # We catch each error type separately so we can give
    # the user a precise, actionable error message.
    logger.debug(
        f"Calling {resolved_model} | "
        f"temp={temperature} | max_tokens={max_tokens}"
    )
    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Extract the text content from the response structure.
        # choices[0]  — take the first (and usually only) completion
        # .message    — the assistant's message object
        # .content    — the actual text string
        raw_content = response.choices[0].message.content

        # Guard against empty responses (rare but possible)
        if not raw_content or not raw_content.strip():
            return LLMError(
                error_type="empty_response",
                message=(
                    f"Model '{resolved_model}' returned an empty response. "
                    f"This can happen with very short inputs or certain model "
                    f"configurations. Try a different model or longer input."
                ),
                retryable=True,
            )

        # Extract token usage — we'll use this in Step 7
        usage = response.usage

        logger.debug(
            f"Response received | "
            f"tokens: {usage.prompt_tokens} in / "
            f"{usage.completion_tokens} out"
        )

        return LLMResponse(
            content=raw_content.strip(),
            model=resolved_model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

    except APIConnectionError as e:
        # Can't reach the server at all.
        # For Ollama: the service isn't running.
        # For OpenAI: no internet connection.
        logger.error(f"Connection failed to {os.getenv('OPENAI_BASE_URL')}: {e}")
        return LLMError(
            error_type="connection",
            message=(
                f"Cannot connect to the LLM server.\n"
                f"  If using Ollama: make sure it's running. "
                f"Check the system tray or run `ollama list`.\n"
                f"  If using OpenAI: check your internet connection.\n"
                f"  Base URL: {os.getenv('OPENAI_BASE_URL', 'http://localhost:11434/v1')}\n"
                f"  Detail: {e}"
            ),
            retryable=False,  # Retrying won't help if the server is down
        )

    except APITimeoutError as e:
        # Server is reachable but took too long to respond.
        # Common with large models on slow hardware.
        logger.warning(f"Timeout after 60s for model '{resolved_model}'")
        return LLMError(
            error_type="timeout",
            message=(
                f"LLM call timed out after 60 seconds.\n"
                f"  The model might be loading for the first time "
                f"(first call is always slower).\n"
                f"  Try again — subsequent calls are faster.\n"
                f"  Or switch to a lighter model: phi3 or gemma2:2b.\n"
                f"  Detail: {e}"
            ),
            retryable=True,  # Worth retrying — model might finish loading
        )

    except RateLimitError as e:
        # Only relevant for paid APIs like OpenAI.
        # Ollama has no rate limits.
        logger.warning(f"Rate limit hit — wait 60s before retrying")
        return LLMError(
            error_type="rate_limit",
            message=(
                f"Rate limit hit. Too many requests sent too quickly.\n"
                f"  Wait 60 seconds and try again.\n"
                f"  Detail: {e}"
            ),
            status_code=429,
            retryable=True,
        )

    except APIStatusError as e:
        # Server responded but with an error HTTP status code.
        # 401 = bad API key, 404 = model not found, 500 = server error
        logger.error(f"API error {e.status_code} from '{resolved_model}': {e.message}")
        return LLMError(
            error_type="api_error",
            message=(
                f"API returned error {e.status_code}.\n"
                f"  Common causes:\n"
                f"  • 401: Invalid API key\n"
                f"  • 404: Model '{resolved_model}' not found. "
                f"Run `ollama list` to see available models.\n"
                f"  • 500: Server-side error. Try again later.\n"
                f"  Detail: {e.message}"
            ),
            status_code=e.status_code,
            retryable=e.status_code >= 500,  # Only server errors are retryable
        )


def call_llm_with_pair(
    pair: PromptPair,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 600,
) -> LLMResult:
    """
    Convenience wrapper: call the LLM using a PromptPair directly.

    This is the function the CLI pipeline will actually use — it accepts
    the structured PromptPair from prompt_builder.py and passes the
    system and user parts to call_llm().

    Args:
        pair:        PromptPair containing system and user prompts
        model:       Override model name (or None to use .env)
        temperature: Creativity dial (0.0-1.0)
        max_tokens:  Hard cap on response length

    Returns:
        LLMResult — either LLMResponse or LLMError
    """
    return call_llm(
        system_prompt=pair.system,
        user_prompt=pair.user,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Convenience test function
# ---------------------------------------------------------------------------

def test_connection(model: str | None = None) -> bool:
    """
    Send a minimal test message to verify the LLM is reachable.

    Useful for startup checks: run this before processing files
    so you find out immediately if Ollama isn't running, rather
    than after waiting through file discovery.

    Args:
        model: Model to test. Defaults to OPENAI_MODEL env var.

    Returns:
        True if connection succeeded, False otherwise.
        Prints a diagnostic message either way.
    """
    resolved_model = model or os.getenv("OPENAI_MODEL") or "phi3"
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

    print(f"Testing connection to: {base_url}")
    print(f"Using model: {resolved_model}")

    start = time.time()

    result = call_llm(
        system_prompt="You are a test assistant. Reply with the greetings message of just one line that you confirm the test connection is working.",
        user_prompt="Reply with one line confirmation message with greetings",
        model=resolved_model,
        max_tokens=10,      # Tiny limit — we only need "OK"
    )

    elapsed = time.time() - start

    if isinstance(result, LLMResponse):
        print(f"Connection successful! ({elapsed:.1f}s)")
        print(f"Model replied: {result.content!r}")
        print(f"Tokens used:   {result.total_tokens}")
        return True
    else:
        print(f"Connection failed: {result.error_type}")
        print(f"Message: {result.message}")
        return False