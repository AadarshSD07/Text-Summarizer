"""
json_parser.py — Parses and validates structured JSON from LLM output.

Single responsibility: given a raw string from the LLM, return either
a validated ArticleSummary dataclass or a structured ParseError.

Pipeline:
    raw string → clean → json.loads → validate schema → ArticleSummary

This module handles ALL the ways an LLM can produce malformed JSON:
  - Markdown code fences (```json ... ```)
  - Preamble text ("Here is your summary: {...")
  - Postamble text ("{...} Let me know if you need changes.")
  - Single quotes instead of double quotes
  - Trailing commas
  - Truncated responses

This module does NOT retry. Retrying is the caller's responsibility.
This module does NOT call the LLM. Calling is llm_client's responsibility.
This module ONLY parses. One job. Done well.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from summarizer.logger_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class ArticleSummary:
    """
    A successfully parsed and validated article summary.

    This is the final product of the entire parsing pipeline.
    If you hold one of these, the data is guaranteed to be:
      - title:     non-empty string
      - bullets:   list of exactly 3 non-empty strings
      - sentiment: exactly one of "positive", "negative", "neutral"
      - source:    the filename it came from
    """
    title: str
    bullets: list[str]
    sentiment: str
    source_file: str = ""          # Populated by the caller, not the LLM
    raw_response: str = ""         # The original LLM text, for debugging


# ---------------------------------------------------------------------------
# Error data structure
# ---------------------------------------------------------------------------

class ErrorKind(Enum):
    """
    Categories of parse failure — each implies a different recovery strategy.

    Why an Enum instead of plain strings?
    - Typo-proof: ErrorKind.NO_JSON vs "no_json" (easy to mistype)
    - Discoverable: IDE shows all valid values
    - Comparable: can use `==` without worrying about case
    """
    NO_JSON        = "no_json"         # Cleaner found nothing JSON-shaped
    SYNTAX_ERROR   = "syntax_error"    # json.loads failed
    MISSING_FIELD  = "missing_field"   # Required key absent
    WRONG_TYPE     = "wrong_type"      # Field exists but wrong Python type
    WRONG_VALUE    = "wrong_value"     # Value not in allowed set
    WRONG_LENGTH   = "wrong_length"    # bullets array not exactly 3 items
    EMPTY_CONTENT  = "empty_content"   # LLM returned empty/whitespace


@dataclass
class ParseError:
    """
    A structured description of why parsing failed.

    Structured errors are more useful than raw exceptions because:
    - The retry logic (Step 8) can decide whether to retry based on error_kind
    - The error message can be fed BACK to the LLM in a retry prompt
    - Logging is consistent and machine-readable
    """
    error_kind: ErrorKind
    message: str                    # Human-readable explanation
    raw_response: str = ""          # The original text that failed
    retryable: bool = True          # Should the caller attempt a retry?
    field_name: str = ""            # Which field caused the error (if applicable)


# The union type for parser return values
ParseResult = ArticleSummary | ParseError

# The only valid sentiment values — defined once, used in validation
VALID_SENTIMENTS = {"positive", "negative", "neutral"}


# ---------------------------------------------------------------------------
# Stage 1: Cleaner — extract JSON from messy LLM output
# ---------------------------------------------------------------------------

def clean_llm_output(raw: str) -> str | None:
    """
    Extract the JSON object from raw LLM output.

    LLMs frequently wrap JSON in extra text. This function strips all of
    that away and returns just the JSON string, or None if nothing
    JSON-shaped can be found.

    Strategies applied in order:
      1. Strip markdown code fences (```json...``` or ```...```)
      2. Find the outermost { ... } block using brace matching
      3. If no braces found at all, return None

    Args:
        raw: The raw string from the LLM response

    Returns:
        A cleaned string containing only the JSON object,
        or None if no JSON-shaped content could be found.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Strategy 1: Strip markdown code fences
    # Pattern: optional ```json or ``` at start, ``` at end
    # re.DOTALL makes . match newlines too (JSON spans multiple lines)
    #
    # Why two patterns? Some models use ```json, others use just ```
    fenced = re.sub(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        r"\1",          # Keep only the content between the fences
        text,
    ).strip()

    # If stripping fences changed the text, work with the stripped version
    if fenced != text:
        text = fenced

    # Strategy 2: Find the outermost { ... } block
    # This handles preamble ("Here is your JSON: {...}")
    # and postamble ("{...}\n\nLet me know if you need changes.")
    #
    # We use brace counting rather than a regex because JSON can contain
    # nested objects, and regex can't reliably match nested structures.
    start = text.find("{")
    if start == -1:
        return None     # No opening brace at all — not JSON

    # Count braces to find the matching closing brace
    depth = 0
    end = -1
    for i, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i
                break   # Found the matching closing brace

    if end == -1:
        return None     # Unmatched brace — truncated response

    return text[start : end + 1]   # Slice out just the JSON object


# ---------------------------------------------------------------------------
# Stage 2: Common fixes for near-valid JSON
# ---------------------------------------------------------------------------

def fix_common_json_errors(text: str) -> str:
    """
    Apply heuristic fixes for JSON that is almost-but-not-quite valid.

    These are ordered from safest (least likely to corrupt valid JSON)
    to most aggressive (more likely to have false positives).

    Only call this AFTER clean_llm_output() has extracted the JSON block.

    Args:
        text: A string believed to contain a JSON object

    Returns:
        The string with common issues fixed. May still not be valid JSON.
    """
    # Fix 1: Trailing commas before closing bracket or brace
    # {"key": "value",}  →  {"key": "value"}
    # ["a", "b", "c",]   →  ["a", "b", "c"]
    #
    # Python's json module rejects trailing commas; JavaScript allows them.
    # LLMs trained on JavaScript code sometimes generate them.
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Fix 2: Normalise smart/curly quotes to straight ASCII quotes
    # "value"  →  "value"    (left/right double quotes)
    # 'value'  →  "value"    (single quotes)
    #
    # Word processors and some locales use curly quotes. JSON requires
    # straight double quotes for both keys and string values.
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # "" → ""
    text = text.replace("\u2018", '"').replace("\u2019", '"')  # '' → ""

    # Fix 3: Lowercase the sentiment value if it came back in wrong case
    # "sentiment": "Positive"  →  "sentiment": "positive"
    #
    # We match just the sentiment field specifically to avoid lowercasing
    # content in other fields (titles should keep their capitalisation).
    text = re.sub(
        r'("sentiment"\s*:\s*)"([^"]+)"',
        lambda m: m.group(1) + '"' + m.group(2).lower() + '"',
        text,
        flags=re.IGNORECASE,
    )

    return text


# ---------------------------------------------------------------------------
# Stage 3: Schema validation
# ---------------------------------------------------------------------------

def validate_summary_schema(data: dict) -> ParseError | None:
    """
    Validate that a parsed dict matches the ArticleSummary schema.

    This is a SEMANTIC check — json.loads already confirmed the syntax.
    Here we check:
      - Are all required fields present?
      - Are they the right Python types?
      - Are values within the allowed set?
      - Is the bullets array exactly 3 items?

    Args:
        data: A Python dict from json.loads()

    Returns:
        None if validation passes (caller can proceed).
        ParseError describing the first violation found.
    """
    # Check 1: All required fields present
    required_fields = {"title", "bullets", "sentiment"}
    missing = required_fields - data.keys()
    if missing:
        missing_str = ", ".join(f'"{f}"' for f in sorted(missing))
        return ParseError(
            error_kind=ErrorKind.MISSING_FIELD,
            message=(
                f"Response is missing required field(s): {missing_str}. "
                f"Fields present: {list(data.keys())}"
            ),
            field_name=", ".join(sorted(missing)),
        )

    # Check 2: title must be a non-empty string
    if not isinstance(data["title"], str):
        return ParseError(
            error_kind=ErrorKind.WRONG_TYPE,
            message=(
                f"'title' must be a string, "
                f"got {type(data['title']).__name__}: {data['title']!r}"
            ),
            field_name="title",
        )
    if not data["title"].strip():
        return ParseError(
            error_kind=ErrorKind.WRONG_VALUE,
            message="'title' must not be empty.",
            field_name="title",
        )

    # Check 3: bullets must be a list
    if not isinstance(data["bullets"], list):
        return ParseError(
            error_kind=ErrorKind.WRONG_TYPE,
            message=(
                f"'bullets' must be a list/array, "
                f"got {type(data['bullets']).__name__}: {data['bullets']!r}"
            ),
            field_name="bullets",
        )

    # Check 4: bullets must contain exactly 3 items
    if len(data["bullets"]) != 3:
        return ParseError(
            error_kind=ErrorKind.WRONG_LENGTH,
            message=(
                f"'bullets' must contain exactly 3 items, "
                f"got {len(data['bullets'])}."
            ),
            field_name="bullets",
        )

    # Check 5: each bullet must be a non-empty string
    for i, bullet in enumerate(data["bullets"]):
        if not isinstance(bullet, str):
            return ParseError(
                error_kind=ErrorKind.WRONG_TYPE,
                message=(
                    f"bullets[{i}] must be a string, "
                    f"got {type(bullet).__name__}: {bullet!r}"
                ),
                field_name=f"bullets[{i}]",
            )
        if not bullet.strip():
            return ParseError(
                error_kind=ErrorKind.WRONG_VALUE,
                message=f"bullets[{i}] must not be empty.",
                field_name=f"bullets[{i}]",
            )

    # Check 6: sentiment must be one of the allowed values
    if not isinstance(data["sentiment"], str):
        return ParseError(
            error_kind=ErrorKind.WRONG_TYPE,
            message=(
                f"'sentiment' must be a string, "
                f"got {type(data['sentiment']).__name__}"
            ),
            field_name="sentiment",
        )
    if data["sentiment"].lower() not in VALID_SENTIMENTS:
        return ParseError(
            error_kind=ErrorKind.WRONG_VALUE,
            message=(
                f"'sentiment' must be one of {sorted(VALID_SENTIMENTS)}, "
                f"got {data['sentiment']!r}."
            ),
            field_name="sentiment",
        )

    # All checks passed
    return None


# ---------------------------------------------------------------------------
# Stage 4: Full parse pipeline — the public API of this module
# ---------------------------------------------------------------------------

def parse_llm_response(
    raw_response: str,
    source_file: str = "",
) -> ParseResult:
    """
    The single entry point for all JSON parsing.

    Runs the full pipeline:
        raw string → clean → json.loads → validate → ArticleSummary

    This is the ONLY function callers should use from this module.
    All other functions are internal pipeline stages.

    Args:
        raw_response: The exact string returned by the LLM API
        source_file:  Filename for context (included in result)

    Returns:
        ArticleSummary if all pipeline stages succeed.
        ParseError describing the first failure encountered.
    """
    # Guard: empty or whitespace-only response
    if not raw_response or not raw_response.strip():
        return ParseError(
            error_kind=ErrorKind.EMPTY_CONTENT,
            message="LLM returned an empty response.",
            raw_response=raw_response,
            retryable=True,
        )

    # Stage 1: Clean — extract JSON from surrounding text
    cleaned = clean_llm_output(raw_response)
    if cleaned is None:
        logger.warning(f"No JSON found in LLM output for '{source_file}'")
        return ParseError(
            error_kind=ErrorKind.NO_JSON,
            message=(
                "Could not find a JSON object in the LLM response. "
                f"Response started with: {raw_response[:100]!r}"
            ),
            raw_response=raw_response,
            retryable=True,
        )

    logger.debug(f"Cleaned output: {cleaned[:80]!r}")

    # Stage 2: Apply common fixes before parsing
    fixed = fix_common_json_errors(cleaned)
    if fixed != cleaned:
        logger.debug(f"Applied JSON fixes for '{source_file}'")

    # Stage 3: Parse — convert JSON string to Python dict
    try:
        data = json.loads(fixed)
    except json.JSONDecodeError as e:
        logger.warning(
            f"JSON syntax error in '{source_file}' at "
            f"line {e.lineno}: {e.msg}"
        )
        return ParseError(
            error_kind=ErrorKind.SYNTAX_ERROR,
            message=(
                f"JSON syntax error at line {e.lineno}, col {e.colno}: {e.msg}\n"
                f"Cleaned text was: {fixed[:200]!r}"
            ),
            raw_response=raw_response,
            retryable=True,
        )

    # Confirm we parsed an object (dict), not an array or primitive
    if not isinstance(data, dict):
        return ParseError(
            error_kind=ErrorKind.WRONG_TYPE,
            message=(
                f"Expected a JSON object {{...}}, "
                f"got {type(data).__name__}: {str(data)[:100]!r}"
            ),
            raw_response=raw_response,
            retryable=True,
        )

    # Stage 4: Validate schema
    validation_error = validate_summary_schema(data)
    if validation_error is not None:
        # Attach the raw response to the error for debugging and retry
        logger.warning(
            f"Schema validation failed for '{source_file}': "
            f"{validation_error.error_kind.value} — {validation_error.message}"
        )
        validation_error.raw_response = raw_response
        return validation_error

    logger.debug(
        f"Parse succeeded for '{source_file}': "
        f"sentiment={data['sentiment']}"
    )

    # All stages passed — build and return the typed dataclass
    return ArticleSummary(
        title=data["title"].strip(),
        bullets=[b.strip() for b in data["bullets"]],
        sentiment=data["sentiment"].lower().strip(),
        source_file=source_file,
        raw_response=raw_response,
    )


# ---------------------------------------------------------------------------
# Output: save results to disk
# ---------------------------------------------------------------------------

def save_summary_to_file(
    summary: ArticleSummary,
    output_dir,           # Path object
) -> None:
    """
    Save a parsed ArticleSummary to a JSON file in output_dir.

    Output filename matches source: article1.txt → article1.json
    The file is formatted with 2-space indentation for readability.

    Args:
        summary:    The validated ArticleSummary to save
        output_dir: Path to the output folder (must exist)
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output filename from source filename
    # article1.txt → article1.json
    source = Path(summary.source_file)
    out_path = output_dir / source.with_suffix(".json").name

    # Build the output dict — clean, no internal fields like raw_response
    output_data = {
        "source_file": summary.source_file,
        "title":       summary.title,
        "bullets":     summary.bullets,
        "sentiment":   summary.sentiment,
    }

    # json.dumps with indent=2 produces human-readable output
    # ensure_ascii=False preserves non-ASCII characters (accents, etc.)
    out_path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"  Saved → {out_path}")