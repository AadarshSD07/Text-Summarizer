"""
retrier.py — Retry engine for LLM calls that produce invalid output.

Single responsibility: given a callable that produces LLMResult,
retry it intelligently when it fails, using correction prompts
and exponential backoff.

Design principles:
  1. Each retry is smarter than the last (correction prompt)
  2. Each retry is more conservative than the last (lower temperature)
  3. Waits grow exponentially to avoid hammering a struggling server
  4. Maximum retry limit is always enforced
  5. Non-retryable errors are surfaced immediately without waiting

This module knows about LLM calls and parsing.
It does NOT know about files, CLI, or JSON output format.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from summarizer.llm_client import (
    call_llm_with_pair,
    LLMResponse,
    LLMError,
    LLMResult,
)
from summarizer.logger_config import get_logger
from summarizer.json_parser import (
    parse_llm_response,
    ArticleSummary,
    ParseError,
    ParseResult,
)
from summarizer.prompt_builder import (
    PromptPair,
    build_correction_prompt,
)
from summarizer.token_counter import check_token_budget

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AttemptRecord:
    """
    A record of one attempt in the retry loop.
    Accumulated for diagnostics and verbose logging.
    """
    attempt_number: int         # 1-indexed
    temperature:    float       # Temperature used for this attempt
    llm_result:     LLMResult   # What the LLM returned (or error)
    parse_result:   ParseResult | None  # What the parser found
    wait_before_s:  float       # How long we waited before this attempt
    succeeded:      bool        # Did this attempt produce a valid summary?


@dataclass
class RetryResult:
    """
    The final outcome of the full retry loop.

    Contains either a successful ArticleSummary or the last error seen,
    plus a complete history of all attempts for diagnostics.
    """
    succeeded:    bool
    summary:      ArticleSummary | None     # Set if succeeded=True
    final_error:  ParseError | LLMError | None  # Set if succeeded=False
    attempts:     list[AttemptRecord] = field(default_factory=list)
    total_tokens: int = 0                   # Summed across all attempts


# ---------------------------------------------------------------------------
# Core retry engine
# ---------------------------------------------------------------------------

def run_with_retry(
    prompt_pair:      PromptPair,
    article_text:     str,
    source_file:      str = "",
    model:            str | None = None,
    max_retries:      int = 3,
    base_temperature: float = 0.3,
    base_delay:       float = 1.0,
    verbose:          bool = False,
) -> RetryResult:
    """
    Run the summarization pipeline with intelligent retry logic.

    Attempt 1: Original prompt at base_temperature
    Attempt 2+: Correction prompt showing the model its mistake,
                at progressively lower temperature for more determinism

    Backoff: wait = base_delay × (2 ^ (attempt - 1))
    So: 0s (attempt 1), 1s (attempt 2), 2s (attempt 3), 4s (attempt 4)

    Args:
        prompt_pair:      Original system+user prompts from prompt_builder
        article_text:     The raw article text (for correction prompts)
        source_file:      Filename for context (included in results)
        model:            Model override (None = use .env)
        max_retries:      Maximum total attempts (1 = no retry, just try once)
        base_temperature: Temperature for attempt 1 (reduced each retry)
        base_delay:       Base seconds to wait between attempts
        verbose:          Print attempt-by-attempt details

    Returns:
        RetryResult with succeeded, summary or error, and attempt history
    """
    attempts: list[AttemptRecord] = []
    current_pair = prompt_pair          # May be replaced with correction prompt
    total_tokens  = 0

    for attempt_num in range(1, max_retries + 1):

        # ── Calculate this attempt's parameters ──────────────────
        # Temperature decreases each attempt for more deterministic output:
        # Attempt 1: base_temperature (e.g. 0.3)
        # Attempt 2: base_temperature / 2 (e.g. 0.15)
        # Attempt 3: 0.1 (minimum — don't go lower than this)
        temperature = max(0.1, base_temperature / (attempt_num))

        # Exponential backoff — wait before all attempts except the first
        # Formula: base_delay × 2^(attempt-1)
        # 0s, 1s, 2s, 4s, 8s, ...
        wait_s = base_delay * (2 ** (attempt_num - 2)) if attempt_num > 1 else 0.0

        # ── Wait (if not first attempt) ───────────────────────────
        if wait_s > 0:
            logger.debug(
                f"[{source_file}] Waiting {wait_s:.0f}s before "
                f"attempt {attempt_num}/{max_retries}"
            )
            if verbose:
                print(
                    f"    [Retry {attempt_num}/{max_retries}] "
                    f"Waiting {wait_s:.0f}s before retry..."
                )
            time.sleep(wait_s)
        elif verbose and attempt_num == 1:
            print(f"    [Attempt {attempt_num}/{max_retries}] temperature={temperature}")

        # ── Call the LLM ──────────────────────────────────────────
        llm_result = call_llm_with_pair(
            pair=current_pair,
            model=model,
            temperature=temperature,
            max_tokens=600,
        )

        # Accumulate token usage across all attempts
        if isinstance(llm_result, LLMResponse):
            total_tokens += llm_result.total_tokens

        # ── Handle LLM-level errors ───────────────────────────────
        if isinstance(llm_result, LLMError):
            record = AttemptRecord(
                attempt_number=attempt_num,
                temperature=temperature,
                llm_result=llm_result,
                parse_result=None,
                wait_before_s=wait_s,
                succeeded=False,
            )
            attempts.append(record)

            logger.error(
                f"[{source_file}] Attempt {attempt_num}: "
                f"LLM error '{llm_result.error_type}'"
            )

            # If the error is not retryable (e.g. bad API key, server down),
            # stop immediately — retrying won't help
            if not llm_result.retryable:
                logger.error(
                    f"[{source_file}] Non-retryable error — stopping"
                )
                return RetryResult(
                    succeeded=False,
                    summary=None,
                    final_error=llm_result,
                    attempts=attempts,
                    total_tokens=total_tokens,
                )

            # Retryable LLM error — continue to next attempt
            # No correction prompt needed (the error was infrastructure,
            # not content — just try again with the same prompt)
            continue

        # ── Parse the LLM response ────────────────────────────────
        parse_result = parse_llm_response(
            raw_response=llm_result.content,
            source_file=source_file,
        )

        record = AttemptRecord(
            attempt_number=attempt_num,
            temperature=temperature,
            llm_result=llm_result,
            parse_result=parse_result,
            wait_before_s=wait_s,
            succeeded=isinstance(parse_result, ArticleSummary),
        )
        attempts.append(record)

        # ── Success ───────────────────────────────────────────────
        if isinstance(parse_result, ArticleSummary):
            if attempt_num == 1:
                logger.info(f"[{source_file}] Succeeded on first attempt")
            else:
                logger.info(
                    f"[{source_file}] Succeeded on attempt "
                    f"{attempt_num}/{max_retries}"
                )
            return RetryResult(
                succeeded=True,
                summary=parse_result,
                final_error=None,
                attempts=attempts,
                total_tokens=total_tokens,
            )

        # ── Parse failure — build correction prompt for next attempt ──
        if isinstance(parse_result, ParseError):
            logger.warning(
                f"[{source_file}] Attempt {attempt_num}: "
                f"parse error '{parse_result.error_kind.value}'"
            )

            # If we have more attempts left, build a correction prompt
            if attempt_num < max_retries:
                current_pair = build_correction_prompt(
                    original_article=article_text,
                    bad_output=llm_result.content,
                    error_message=parse_result.message,
                    error_kind=parse_result.error_kind.value,
                    attempt_number=attempt_num,
                )
                logger.debug(
                    f"[{source_file}] Building correction prompt for "
                    f"error: {parse_result.message[:80]}"
                )

    # ── All attempts exhausted ────────────────────────────────────
    # Return the last error we saw
    last_parse_error = next(
        (a.parse_result for a in reversed(attempts)
         if isinstance(a.parse_result, ParseError)),
        None,
    )
    last_llm_error = next(
        (a.llm_result for a in reversed(attempts)
         if isinstance(a.llm_result, LLMError)),
        None,
    )

    logger.error(
        f"[{source_file}] All {max_retries} attempts failed. "
        f"Last error: "
        f"{(last_parse_error or last_llm_error)}"
    )

    return RetryResult(
        succeeded=False,
        summary=None,
        final_error=last_parse_error or last_llm_error,
        attempts=attempts,
        total_tokens=total_tokens,
    )


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def format_retry_report(result: RetryResult, filename: str = "") -> str:
    """
    Format a human-readable summary of all retry attempts.

    Call this in verbose mode to see exactly what happened across
    every attempt — useful for debugging prompt issues.

    Args:
        result:   The RetryResult from run_with_retry()
        filename: Optional filename label

    Returns:
        Multi-line string describing every attempt
    """
    label = f" ({filename})" if filename else ""
    lines = [f"Retry report{label}: {len(result.attempts)} attempt(s)"]

    for rec in result.attempts:
        status = "OK" if rec.succeeded else "FAIL"
        lines.append(
            f"  Attempt {rec.attempt_number}: [{status}] "
            f"temp={rec.temperature:.2f}"
            + (f"  waited {rec.wait_before_s:.0f}s" if rec.wait_before_s else "")
        )

        if isinstance(rec.llm_result, LLMResponse):
            lines.append(
                f"    LLM: {rec.llm_result.total_tokens} tokens total"
            )

        if isinstance(rec.parse_result, ParseError):
            lines.append(
                f"    Error: {rec.parse_result.error_kind.value} — "
                f"{rec.parse_result.message[:100]}"
            )
        elif isinstance(rec.llm_result, LLMError):
            lines.append(
                f"    LLM Error: {rec.llm_result.error_type} — "
                f"{rec.llm_result.message[:100]}"
            )

    lines.append(
        f"  Final: {'SUCCESS' if result.succeeded else 'FAILED'} "
        f"| Total tokens: {result.total_tokens}"
    )
    return "\n".join(lines)