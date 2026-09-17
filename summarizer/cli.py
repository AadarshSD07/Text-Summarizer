"""
cli.py — Entry point for the text-summarizer tool.

This module is responsible for ONE thing only:
  1. Parsing command-line arguments
  2. Calling the appropriate functions with those arguments
  3. Reporting success or failure to the user

It does NOT contain business logic. It delegates to other modules.
This separation means you can test your business logic without
ever touching the CLI layer.
"""
import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from pathlib import Path
from summarizer.file_reader import read_txt_files, FileContent, FileError
from summarizer.json_parser import parse_llm_response, ArticleSummary, ParseError, save_summary_to_file
from summarizer.logger_config import setup_logging, get_logger
from summarizer.llm_client import call_llm_with_pair, call_llm, test_connection, LLMResponse, LLMError
from summarizer.prompt_builder import build_summary_prompt
from summarizer.retrier import run_with_retry, format_retry_report
from summarizer.token_counter import (
    check_token_budget,
    truncate_to_token_limit,
    format_budget_report,
    UsageRecord,
    SessionUsage,
)

logger = get_logger(__name__)

# We'll import these as we build them in later steps.
# For now, we stub them so the CLI can be tested independently.
# from summarizer.file_reader import read_text_files
# from summarizer.llm_client import summarize_file
# from summarizer.json_parser import save_results


def build_parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser.

    Why a separate function instead of inline in main()?
    - Easier to test the parser independently
    - Easier to read: parser construction is isolated from execution
    - Follows the single responsibility principle

    Returns:
        argparse.ArgumentParser: Configured parser ready to parse args.
    """
    parser = argparse.ArgumentParser(
        # The name shown in --help as the command name
        prog="summarize",

        # A short description shown at the top of --help
        description=(
            "Read .txt files from a folder and summarize each one "
            "using a local LLM. Outputs structured JSON with a title, "
            "three bullet points, and a sentiment score."
        ),

        # Text shown at the bottom of --help — good place for examples
        epilog=(
            "Examples:\n"
            "  summarize --input texts/\n"
            "  summarize --input texts/ --output results/ --verbose\n"
            "  summarize --input texts/ --model phi3\n"
        ),

        # Preserves newlines in epilog (default formatter collapses them)
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input/Output arguments ---

    parser.add_argument(
        "--input",
        "-i",                          # Short form: -i texts/
        type=Path,                     # Converts string → pathlib.Path object automatically
        default=Path("texts"),         # If not specified, uses ./texts/
        metavar="FOLDER",             # Shows FOLDER instead of INPUT in --help
        help="Folder containing .txt files to summarize (default: ./texts)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        metavar="FOLDER",
        help="Folder where JSON results will be saved (default: ./output)",
    )

    # --- Behaviour arguments ---

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,                  # None means: read from .env file
        metavar="MODEL_NAME",
        help=(
            "LLM model to use, e.g. mistral, phi3, gpt-4o-mini. "
            "Overrides the OPENAI_MODEL value in your .env file."
        ),
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",           # No value needed — presence = True
        help="Print detailed progress information to the terminal.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Read and display files but do NOT call the LLM. "
            "Useful for testing your file setup without using any tokens."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help=(
            "Console log level (default: WARNING). "
            "File always captures DEBUG and above. "
            "Use DEBUG to see every internal step."
        ),
    )

    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test that the LLM server is reachable, then exit.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> list[str]:
    """
    Check that the parsed arguments make sense together.

    argparse handles syntax (--input requires a value, --verbose is boolean).
    This function handles SEMANTICS (does the input folder actually exist?).

    Why separate from argparse?
    argparse can validate types but not business rules like
    "the folder must exist" or "the folder must contain .txt files".
    We check those here and return human-friendly error messages.

    Args:
        args: The namespace object returned by parse_args()

    Returns:
        list[str]: List of error messages. Empty list = all good.
    """
    errors = []

    # Check input folder exists
    if not args.input.exists():
        errors.append(
            f"Input folder not found: '{args.input}'\n"
            f"  Create it with: mkdir -p {args.input}\n"
            f"  Then add some .txt files to it."
        )
    elif not args.input.is_dir():
        errors.append(
            f"'{args.input}' is a file, not a folder. "
            f"Please provide a folder path."
        )
    else:
        # Folder exists — check it has .txt files
        txt_files = list(args.input.glob("*.txt"))
        if not txt_files:
            errors.append(
                f"No .txt files found in '{args.input}'\n"
                f"  Add some text files and try again."
            )

    return errors


def main() -> None:
    """
    Main entry point called by the `summarize` CLI command.

    Flow:
        1. Parse arguments from sys.argv
        2. Validate that arguments make semantic sense
        3. Create output directory if needed
        4. Run the summarization pipeline (Steps 3–9)
        5. Report results

    This function should stay thin — its job is orchestration,
    not implementation. Each step delegates to a specialist module.
    """
    # Load .env FIRST — before reading any env vars anywhere
    load_dotenv()

    # Step 1: Parse arguments
    # parse_args() reads sys.argv[1:] by default
    # (sys.argv[0] is the program name itself, which we skip)
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(
        log_dir=Path("logs"),
        console_level=getattr(logging, args.log_level),
        verbose=args.verbose,
    )

    logger.info(
        f"Run started | input={args.input} | "
        f"output={args.output} | model={args.model or 'from .env'}"
    )

    # Step 2: Validate semantic correctness
    errors = validate_args(args)
    if errors:
        # Print each error, then show usage hint and exit
        print("Error: Invalid arguments\n", file=sys.stderr)
        for error in errors:
            print(f"  • {error}", file=sys.stderr)
        print(f"\nRun `summarize --help` for usage information.", file=sys.stderr)
        sys.exit(1)          # Exit code 1 = failure (convention)

    if args.test_connection:
        success = test_connection(model=args.model)
        sys.exit(0 if success else 1)

    # Step 3: Create output directory if it doesn't exist
    # parents=True: creates intermediate folders too (like mkdir -p)
    # exist_ok=True: no error if it already exists
    args.output.mkdir(parents=True, exist_ok=True)

    # Step 4: Show what we're about to do (if verbose)
    if args.verbose:
        txt_files = list(args.input.glob("*.txt"))
        print(f"Input folder:  {args.input.resolve()}")
        print(f"Output folder: {args.output.resolve()}")
        print(f"Files found:   {len(txt_files)}")
        print(f"Model:         {args.model or 'from .env'}")
        print(f"Dry run:       {args.dry_run}")
        print()

    # Step 5: Run the pipeline — stubbed for now, will expand in later steps
    # if args.dry_run:
    #     txt_files = list(args.input.glob("*.txt"))
    #     print(f"[Dry run] Would summarize {len(txt_files)} file(s):")
    #     for f in sorted(txt_files):
    #         print(f"  • {f.name}")
    #     print("\n[Dry run] No LLM calls made.")
    #     sys.exit(0)
    if args.dry_run:
        print("Dry run — scanning files only, no LLM calls will be made.")
        print()

        found_any = False
        for result in read_txt_files(args.input, max_files=getattr(args, 'max_files', None)):
            found_any = True
            if isinstance(result, FileContent):
                print(f"  FOUND  {result.filename}  ({result.char_count} chars, {result.encoding_used})")
            elif isinstance(result, FileError):
                print(f"  ERROR  {result.filename}: {result.error_type}")

        if not found_any:
            print("  No .txt files found.")

        sys.exit(0)

    # ── Full pipeline ────────────────────────────────────────────────
    print(f"Reading files from: {args.input}")
    print(f"Saving results to:  {args.output}")
    print()

    succeeded = []
    failed    = []

    session_usage = SessionUsage()

    for file_result in read_txt_files(args.input):

        # Skip files that couldn't be read
        if isinstance(file_result, FileError):
            print(f"  SKIP  {file_result.filename} — {file_result.message}")
            failed.append(file_result.filename)
            continue

        print("─" * 60)
        print(f"  Processing {file_result.filename}...", end=" ", flush=True)

        resolved_model = args.model or os.getenv("OPENAI_MODEL") or "mistral"
        max_retries    = int(os.getenv("MAX_RETRIES", "3"))

        # Build prompts
        pair = build_summary_prompt(
            article_text=file_result.content,
            filename=file_result.filename,
        )

        # ── PRE-FLIGHT: check token budget ───────────────────────
        budget = check_token_budget(
            system_prompt=pair.system,
            user_prompt=pair.user,
            model=resolved_model,
        )

        if args.verbose:
            print()   # newline before budget report
            print(format_budget_report(budget))

        # If input is too large, truncate the article content and rebuild
        if not budget.is_safe:
            max_article_tokens = (
                budget.max_safe_input - budget.system_tokens - 50
            )
            truncated_content, was_truncated = truncate_to_token_limit(
                text=file_result.content,
                max_tokens=max_article_tokens,
                model=resolved_model,
            )

            if was_truncated:
                if args.verbose:
                    print(
                        f"    Truncated article from "
                        f"~{budget.user_tokens} to "
                        f"~{max_article_tokens} tokens"
                    )
                # Rebuild prompts with truncated content
                pair = build_summary_prompt(
                    article_text=truncated_content,
                    filename=file_result.filename,
                )

        retry_result = run_with_retry(
            prompt_pair=pair,
            article_text=file_result.content,
            source_file=file_result.filename,
            model=args.model,
            max_retries=max_retries,
            verbose=args.verbose,
        )

        # Show full retry report in verbose mode
        if args.verbose:
            print(format_retry_report(retry_result, file_result.filename))

        if not retry_result.succeeded:
            err = retry_result.final_error
            if isinstance(err, ParseError):
                print(f"FAILED (parse: {err.error_kind.value})")
            elif isinstance(err, LLMError):
                print(f"FAILED (llm: {err.error_type})")
            else:
                print("FAILED (unknown error)")
            failed.append(file_result.filename)
            continue

        # Accumulate token usage from all retry attempts
        session_usage.add(UsageRecord(
            filename=file_result.filename,
            prompt_tokens=sum(
                a.llm_result.prompt_tokens
                for a in retry_result.attempts
                if isinstance(a.llm_result, LLMResponse)
            ),
            completion_tokens=sum(
                a.llm_result.completion_tokens
                for a in retry_result.attempts
                if isinstance(a.llm_result, LLMResponse)
            ),
            total_tokens=retry_result.total_tokens,
        ))

        # Save successful result
        save_summary_to_file(retry_result.summary, args.output)

        if args.verbose:
            print(f"    Title:     {retry_result.summary.title}")
            print(f"    Sentiment: {retry_result.summary.sentiment}")
        else:
            attempts_label = (
                f" (after {len(retry_result.attempts)} attempts)"
                if len(retry_result.attempts) > 1 else ""
            )
            print(f"OK{attempts_label}")

        succeeded.append(file_result.filename)

    # ── Final report ─────────────────────────────────────────────────
    print()
    print(f"Done. {len(succeeded)} succeeded, {len(failed)} failed.")

    # Always print token usage — it's not verbose-only
    if session_usage.file_count > 0:
        print()
        print(session_usage.report())

    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  • {f}")

    logger.info(
        f"Run complete | succeeded={len(succeeded)} | "
        f"failed={len(failed)} | "
        f"total_tokens={session_usage.total_tokens}"
    )

    sys.exit(0 if not failed else 1)

# This guard is essential:
# When Python imports a module, it executes all top-level code.
# Without this guard, just doing `from summarizer.cli import build_parser`
# would immediately run main() — which is never what you want.
#
# With this guard, main() only runs when you execute this file directly:
#   python summarizer/cli.py     → runs main()
#   import summarizer.cli        → does NOT run main()
if __name__ == "__main__":
    main()