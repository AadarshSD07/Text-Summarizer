"""
prompt_builder.py — Constructs system and user prompts for the LLM.

Single responsibility: given a piece of text (an article), produce
the exact system prompt and user prompt needed for the LLM to return
a consistent, structured JSON summary.

This module is intentionally separated from llm_client.py because:
  - Prompts change frequently during development and experimentation
  - You want to test prompts independently without making API calls
  - Different use cases may need different prompt styles
  - Prompt logic is business logic — not infrastructure

Nothing in here makes API calls. It only builds strings.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data structure for a prompt pair
# ---------------------------------------------------------------------------

@dataclass
class PromptPair:
    """
    A matched pair of system and user prompts ready to send to the LLM.

    Keeping them together as a named pair (rather than returning a tuple)
    makes the calling code self-documenting:

        pair = build_summary_prompt(text)
        result = call_llm(pair.system, pair.user)   # crystal clear

    rather than:

        system, user = build_summary_prompt(text)   # easy to swap order
    """
    system: str    # The system prompt — role, rules, format specification
    user: str      # The user prompt — the actual article content


# ---------------------------------------------------------------------------
# The target JSON schema — defined once, used in both system prompt and docs
# ---------------------------------------------------------------------------

# This is the exact JSON structure we want the model to produce.
# Defining it as a constant means the system prompt and any validation
# code in json_parser.py can both reference the same truth.
EXPECTED_SCHEMA = {
    "title": "string — a concise, descriptive title for the article (5-10 words)",
    "bullets": [
        "string — key point 1 (one complete sentence)",
        "string — key point 2 (one complete sentence)",
        "string — key point 3 (one complete sentence)",
    ],
    "sentiment": "string — exactly one of: positive, negative, neutral",
}

# The few-shot example — a perfect model response for the model to pattern-match
FEW_SHOT_EXAMPLE = '''{
  "title": "Scientists Discover Water Ice Beneath Mars South Pole",
  "bullets": [
    "Radar data from ESA's Mars Express confirmed a 20km lake of liquid water beneath the Martian south pole.",
    "The discovery is significant because liquid water is considered a prerequisite for life as we know it.",
    "Further missions are being planned to investigate whether microbial life could survive in these conditions."
  ],
  "sentiment": "positive"
}'''


# ---------------------------------------------------------------------------
# Core prompt builder
# ---------------------------------------------------------------------------

def build_summary_prompt(
    article_text: str,
    filename: str = "unknown",
    include_example: bool = True,
) -> PromptPair:
    """
    Build the system and user prompts for summarizing one article.

    This function embodies all prompt engineering decisions:
      - Role definition (who is the model?)
      - Task definition (what must it do?)
      - Format specification (exactly what JSON structure?)
      - Constraints (what must it NOT do?)
      - Few-shot example (what does perfect look like?)

    Args:
        article_text:    The full text content of the article to summarize.
        filename:        The source filename — included in user prompt for context.
        include_example: Whether to include the few-shot example in the system prompt.
                         Set to False if you're close to the context limit.

    Returns:
        PromptPair with system and user prompts ready to send to the LLM.
    """

    # ── SYSTEM PROMPT ────────────────────────────────────────────────────
    # This is the most important string in the entire project.
    # Every word is deliberate. Let's build it in layers.

    # Layer 1: Role — establishes who the model is
    role_section = (
        "You are an expert document analyst and summarization specialist. "
        "Your sole function is to read articles and produce structured JSON summaries. "
        "You are precise, objective, and format-disciplined."
    )

    # Layer 2: Task — what it must do
    task_section = (
        "When given an article, you must produce a JSON object containing:\n"
        "  1. A concise title (5-10 words) that captures the article's main topic\n"
        "  2. Exactly 3 bullet points summarizing the most important information\n"
        "  3. A sentiment classification: the overall tone of the article"
    )

    # Layer 3: Format specification — the exact JSON schema
    format_section = (
        "You must respond with ONLY a raw JSON object. Nothing else.\n\n"
        "Required JSON structure:\n"
        "{\n"
        '  "title": "<5-10 word descriptive title>",\n'
        '  "bullets": [\n'
        '    "<complete sentence — key point 1>",\n'
        '    "<complete sentence — key point 2>",\n'
        '    "<complete sentence — key point 3>"\n'
        "  ],\n"
        '  "sentiment": "<positive|negative|neutral>"\n'
        "}"
    )

    # Layer 4: Constraints — guardrails against unwanted behaviour
    # Each constraint closes a specific failure mode we've observed in testing
    constraints_section = (
        "STRICT RULES — you must follow all of these:\n"
        "• Output ONLY the JSON object. No greeting, no explanation, no commentary.\n"
        "• Do NOT wrap the JSON in markdown code blocks (no ```json or ```).\n"
        "• The 'bullets' array must contain EXACTLY 3 strings, no more, no fewer.\n"
        "• The 'sentiment' field must be EXACTLY one of: positive, negative, neutral.\n"
        "• Each bullet must be a complete sentence ending with a full stop.\n"
        "• Base the summary ONLY on information present in the article.\n"
        "• Do NOT add personal opinions or external knowledge.\n"
        "• Do NOT invent or assume information not stated in the article."
    )

    # Layer 5: Few-shot example — show the model what perfect looks like
    example_section = ""
    if include_example:
        example_section = (
            "Here is an example of a PERFECT response. "
            "Your output must follow this exact structure:\n\n"
            f"{FEW_SHOT_EXAMPLE}"
        )

    # Assemble the full system prompt with clear section breaks
    # The double newlines between sections help the model parse structure
    system_prompt_parts = [
        role_section,
        task_section,
        format_section,
        constraints_section,
    ]

    if example_section:
        system_prompt_parts.append(example_section)

    system_prompt = "\n\n".join(system_prompt_parts)

    # ── USER PROMPT ──────────────────────────────────────────────────────
    # Keep this simple and clean.
    # The system prompt contains all the rules — no need to repeat them.
    # The user prompt's only job is to deliver the article text.
    #
    # We include the filename so the model has context about the source,
    # which can sometimes help with title generation.

    user_prompt = (
        f"Please summarize the following article.\n"
        f"Source file: {filename}\n\n"
        f"ARTICLE:\n"
        f"{article_text}"
    )

    return PromptPair(system=system_prompt, user=user_prompt)


def build_correction_prompt(
    original_article: str,
    bad_output: str,
    error_message: str,
    error_kind: str,
    attempt_number: int,
) -> PromptPair:
    """
    Build a correction prompt for a retry after a parse failure.

    The correction prompt shows the model:
      1. What article it was summarizing
      2. What it returned that was wrong
      3. Precisely what the error was
      4. An explicit instruction to fix only that error

    This is far more effective than re-sending the original prompt,
    because the model now has evidence of its own failure.

    Args:
        original_article: The article text from the original request
        bad_output:       The malformed string the model returned
        error_message:    Human-readable description of what was wrong
        error_kind:       The ErrorKind enum value as a string
        attempt_number:   Which retry this is (1-indexed)

    Returns:
        PromptPair ready to send as a corrected retry
    """
    # The system prompt stays the same — it still describes the job
    # and the required JSON format. We don't need to change it.
    original_pair = build_summary_prompt(
        article_text=original_article,
        filename="",
        include_example=True,   # Include example on retries — model needs reminding
    )

    # The user prompt now carries the evidence of failure
    user_prompt = (
        f"You were asked to summarize an article as a JSON object.\n"
        f"Your previous attempt (attempt {attempt_number}) had an error.\n\n"
        f"═══ YOUR PREVIOUS OUTPUT (which had an error) ═══\n"
        f"{bad_output}\n\n"
        f"═══ THE ERROR FOUND IN YOUR OUTPUT ═══\n"
        f"Error type: {error_kind}\n"
        f"Detail: {error_message}\n\n"
        f"═══ WHAT TO DO ═══\n"
        f"Return a corrected JSON object that fixes this specific error.\n"
        f"Output ONLY the raw JSON. No explanation. No markdown. No preamble.\n\n"
        f"═══ ORIGINAL ARTICLE (summarize this) ═══\n"
        f"{original_article}"
    )

    return PromptPair(
        system=original_pair.system,
        user=user_prompt,
    )


# ---------------------------------------------------------------------------
# Prompt introspection helpers — useful for debugging and Step 7 (tokens)
# ---------------------------------------------------------------------------

def estimate_prompt_length(pair: PromptPair) -> dict[str, int]:
    """
    Return character counts for each part of a prompt pair.

    Useful for:
    - Debugging: seeing how long your prompts actually are
    - Token estimation: chars ÷ 4 ≈ tokens (rough rule of thumb)
    - Context limit checks: make sure you're not exceeding the model's limit

    Args:
        pair: The PromptPair to measure

    Returns:
        Dict with character counts for system, user, and total
    """
    system_chars = len(pair.system)
    user_chars   = len(pair.user)

    return {
        "system_chars":    system_chars,
        "user_chars":      user_chars,
        "total_chars":     system_chars + user_chars,
        "approx_tokens":   (system_chars + user_chars) // 4,  # rough estimate
    }


def print_prompt_debug(pair: PromptPair, filename: str = "") -> None:
    """
    Pretty-print a prompt pair for debugging purposes.

    Call this during development to see exactly what you're sending
    to the model. Indispensable for diagnosing unexpected output.

    Args:
        pair:     The PromptPair to display
        filename: Optional filename label for context
    """
    label = f" ({filename})" if filename else ""
    separator = "─" * 60

    print(f"\n{separator}")
    print(f"PROMPT DEBUG{label}")
    print(separator)
    print("[ SYSTEM PROMPT ]")
    print(pair.system)
    print(f"\n[ USER PROMPT ]")
    print(pair.user)
    lengths = estimate_prompt_length(pair)
    print(f"\n[ LENGTHS ]")
    print(f"  System : {lengths['system_chars']:,} chars")
    print(f"  User   : {lengths['user_chars']:,} chars")
    print(f"  Total  : {lengths['total_chars']:,} chars")
    print(f"  ~Tokens: {lengths['approx_tokens']:,}")
    print(separator)