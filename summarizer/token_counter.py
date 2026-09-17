"""
token_counter.py — Token counting, budget checking, and usage tracking.

Two responsibilities:
  1. PRE-FLIGHT:  count tokens before sending to the LLM so we can
                  warn about context overflow and avoid wasted API calls
  2. POST-FLIGHT: accumulate exact usage from API responses for reporting

Why a separate module?
  Token logic involves model-specific knowledge (context windows, costs)
  that doesn't belong in the LLM client or the CLI. Isolating it here
  means you can change tokenizer libraries or add new models in one place.

Key distinction:
  tiktoken gives ESTIMATES (fast, no API call needed).
  response.usage gives EXACT counts (only available after the call).
  We use estimates for pre-flight guards, exact counts for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import tiktoken


# ---------------------------------------------------------------------------
# Model context window registry
# ---------------------------------------------------------------------------

# Maps model name patterns to their context window sizes (in tokens).
# This is used to calculate the safe input limit per model.
# Source: official model documentation as of mid-2025.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Ollama local models
    "mistral":       8_192,
    "mistral:7b":    8_192,
    "phi3":          8_192,   # Phi-3 Mini 4K context (conservative)
    "phi3:mini":     8_192,
    "gemma2":       16_384,
    "gemma2:2b":    16_384,
    "llama3":        8_192,
    "llama3:8b":     8_192,
    "llama3.1":    131_072,   # Llama 3.1 has 128K context
    "llama3.2":    131_072,

    # OpenAI models (if you switch later)
    "gpt-4o":       128_000,
    "gpt-4o-mini":  128_000,
    "gpt-4-turbo":  128_000,
    "gpt-3.5-turbo": 16_385,

    # Default fallback — conservative for unknown models
    "default":       8_192,
}

# Tokens we always reserve for the model's response.
# Must match the max_tokens argument in call_llm().
RESPONSE_TOKEN_RESERVE = 600

# Additional safety buffer — accounts for tokenizer approximation
# and any formatting tokens the API adds internally.
SAFETY_BUFFER = 200


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TokenBudget:
    """
    A complete picture of the token budget for one API call.

    Created before the call (pre-flight) to decide whether to proceed.
    Also used to generate the verbose token report in the CLI.
    """
    model: str

    # Counts for each prompt component
    system_tokens:  int
    user_tokens:    int
    total_input_tokens: int    # system + user

    # Window calculations
    context_window: int        # Total tokens this model supports
    response_reserve: int      # Tokens reserved for the model's reply
    safety_buffer: int         # Extra breathing room
    max_safe_input: int        # = context_window - reserve - buffer

    # Status
    is_safe: bool              # True if total_input_tokens <= max_safe_input
    overflow_by: int           # How many tokens over budget (0 if safe)


@dataclass
class UsageRecord:
    """
    Exact token usage for one completed API call.

    Populated from response.usage after the call returns.
    More accurate than pre-flight estimates.
    """
    filename: str
    prompt_tokens: int       # Tokens in the input (exact, from API)
    completion_tokens: int   # Tokens in the response (exact, from API)
    total_tokens: int        # Sum of above


@dataclass
class SessionUsage:
    """
    Accumulated token usage across all files in one CLI run.

    Call .add() after each successful API call.
    Call .report() at the end to print a summary.
    """
    records: list[UsageRecord] = field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        """Add one file's usage to the session total."""
        self.records.append(record)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    @property
    def file_count(self) -> int:
        return len(self.records)

    def report(self) -> str:
        """
        Build a human-readable token usage summary string.
        Call this at the end of a CLI run for the final report.
        """
        if not self.records:
            return "No API calls were made."

        lines = [
            "Token usage summary",
            "─" * 40,
        ]

        # Per-file breakdown
        for rec in self.records:
            lines.append(
                f"  {rec.filename:<25} "
                f"{rec.prompt_tokens:>5} in  "
                f"{rec.completion_tokens:>4} out  "
                f"{rec.total_tokens:>5} total"
            )

        lines.append("─" * 40)
        lines.append(
            f"  {'TOTAL':<25} "
            f"{self.total_prompt_tokens:>5} in  "
            f"{self.total_completion_tokens:>4} out  "
            f"{self.total_tokens:>5} total"
        )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core counting functions
# ---------------------------------------------------------------------------

def get_encoder(model: str) -> tiktoken.Encoding:
    """
    Get the appropriate tiktoken encoder for a model.

    tiktoken natively supports OpenAI model names.
    For Ollama models (mistral, phi3, etc.) we fall back to cl100k_base,
    which is close enough for budget estimation purposes.

    Args:
        model: Model name string (e.g. "mistral", "gpt-4o-mini")

    Returns:
        tiktoken.Encoding ready to use for counting
    """
    try:
        # This works for OpenAI model names like "gpt-4o", "gpt-4o-mini"
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Ollama models aren't in tiktoken's registry.
        # cl100k_base is the GPT-4 encoding — a good general approximation.
        # The error is intentionally suppressed: this fallback is expected
        # behaviour for every local model, not a problem to report.
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text string.

    This is an ESTIMATE for local models (within ~5-10%).
    For OpenAI models it is exact.

    Args:
        text:  The string to count tokens for
        model: Model name — used to select the right encoder

    Returns:
        Integer token count
    """
    if not text:
        return 0

    encoder = get_encoder(model)
    return len(encoder.encode(text))


def get_context_window(model: str) -> int:
    """
    Look up the context window size for a given model.

    Checks MODEL_CONTEXT_WINDOWS for an exact match first,
    then falls back to "default" (8,192) for unknown models.

    Args:
        model: Model name string

    Returns:
        Context window size in tokens
    """
    # Try exact match first
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]

    # Try prefix match — e.g. "mistral:latest" matches "mistral"
    for known_model, window in MODEL_CONTEXT_WINDOWS.items():
        if model.lower().startswith(known_model.lower()):
            return window

    # Unknown model — use conservative default
    return MODEL_CONTEXT_WINDOWS["default"]


def check_token_budget(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int = RESPONSE_TOKEN_RESERVE,
) -> TokenBudget:
    """
    Calculate the full token budget for an API call before sending it.

    This is the pre-flight check. Call it before call_llm() to verify
    that your input fits within the model's context window.

    Args:
        system_prompt: The system prompt string
        user_prompt:   The user prompt string (includes article text)
        model:         The model name (determines context window)
        max_tokens:    Tokens reserved for the response

    Returns:
        TokenBudget with all counts and safety status
    """
    system_count = count_tokens(system_prompt, model)
    user_count   = count_tokens(user_prompt,   model)
    total_input  = system_count + user_count

    context_window  = get_context_window(model)
    max_safe_input  = context_window - max_tokens - SAFETY_BUFFER
    is_safe         = total_input <= max_safe_input
    overflow_by     = max(0, total_input - max_safe_input)

    return TokenBudget(
        model=model,
        system_tokens=system_count,
        user_tokens=user_count,
        total_input_tokens=total_input,
        context_window=context_window,
        response_reserve=max_tokens,
        safety_buffer=SAFETY_BUFFER,
        max_safe_input=max_safe_input,
        is_safe=is_safe,
        overflow_by=overflow_by,
    )


def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    model: str = "gpt-4o",
) -> tuple[str, bool]:
    """
    Truncate text to fit within a token limit.

    Used when an article is too long to fit in the context window.
    Truncation is done at the token level for precision, then we
    snap back to the nearest sentence boundary so the text ends
    cleanly rather than mid-word or mid-sentence.

    Args:
        text:       The text to potentially truncate
        max_tokens: Maximum number of tokens allowed
        model:      Model name for the encoder

    Returns:
        Tuple of (possibly-truncated text, was_truncated: bool)
    """
    encoder    = get_encoder(model)
    token_ids  = encoder.encode(text)

    if len(token_ids) <= max_tokens:
        # Text fits — no truncation needed
        return text, False

    # Truncate token IDs to the limit
    truncated_ids  = token_ids[:max_tokens]
    truncated_text = encoder.decode(truncated_ids)

    # Snap to the last sentence boundary so the text ends cleanly.
    # Look for the last period, exclamation, or question mark.
    last_sentence_end = max(
        truncated_text.rfind("."),
        truncated_text.rfind("!"),
        truncated_text.rfind("?"),
    )

    if last_sentence_end > len(truncated_text) * 0.5:
        # Only snap to sentence boundary if we found one in the
        # latter half of the text (don't lose too much content)
        truncated_text = truncated_text[: last_sentence_end + 1]

    return truncated_text.strip(), True


def format_budget_report(budget: TokenBudget) -> str:
    """
    Format a TokenBudget as a human-readable string for verbose output.

    Args:
        budget: The TokenBudget to format

    Returns:
        A multi-line string ready to print
    """
    status = "OK" if budget.is_safe else f"OVERFLOW by {budget.overflow_by:,} tokens"
    used_pct = (budget.total_input_tokens / budget.context_window) * 100

    lines = [
        f"  Token budget ({budget.model})",
        f"    System prompt : {budget.system_tokens:>6,} tokens",
        f"    User prompt   : {budget.user_tokens:>6,} tokens",
        f"    Total input   : {budget.total_input_tokens:>6,} tokens  "
        f"({used_pct:.1f}% of {budget.context_window:,})",
        f"    Max safe input: {budget.max_safe_input:>6,} tokens",
        f"    Status        : {status}",
    ]

    if not budget.is_safe:
        lines.append(
            f"    WARNING: Input exceeds safe limit by {budget.overflow_by:,} tokens."
            f" Article will be truncated."
        )

    return "\n".join(lines)