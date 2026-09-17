"""
Tests for summarizer.token_counter

Token counting is safety-critical — an off-by-large-amount error
could cause context overflow on every call. These tests verify
the counting logic, budget calculation, and truncation.
"""

import pytest
from summarizer.token_counter import (
    count_tokens,
    get_context_window,
    check_token_budget,
    truncate_to_token_limit,
    SessionUsage,
    UsageRecord,
    MODEL_CONTEXT_WINDOWS,
)


class TestCountTokens:
    """Tests for count_tokens()"""

    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_short_text_returns_positive_count(self):
        count = count_tokens("Hello world.")
        assert count > 0

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("Hello.")
        long  = count_tokens("Hello world, this is a much longer sentence.")
        assert long > short

    def test_token_count_approximately_four_chars_per_token(self):
        # English text: 1 token ≈ 4 characters
        text  = "The quick brown fox jumps over the lazy dog."
        count = count_tokens(text)
        chars = len(text)
        ratio = chars / count
        # Ratio should be between 2.5 and 6.0 for normal English
        assert 2.5 <= ratio <= 6.0, (
            f"Unexpected chars/token ratio: {ratio:.1f} "
            f"({chars} chars, {count} tokens)"
        )


class TestGetContextWindow:
    """Tests for get_context_window()"""

    def test_known_model_returns_correct_window(self):
        assert get_context_window("mistral") == 8_192
        assert get_context_window("gpt-4o") == 128_000

    def test_unknown_model_returns_default(self):
        window = get_context_window("some-unknown-model-xyz")
        assert window == MODEL_CONTEXT_WINDOWS["default"]

    def test_tagged_model_matches_base(self):
        # "mistral:latest" should match "mistral" entry
        window = get_context_window("mistral:latest")
        assert window == MODEL_CONTEXT_WINDOWS["mistral"]

    def test_all_registered_models_have_positive_window(self):
        for model, window in MODEL_CONTEXT_WINDOWS.items():
            assert window > 0, f"Model '{model}' has non-positive window"


class TestCheckTokenBudget:
    """Tests for check_token_budget()"""

    def test_short_prompts_are_safe(self):
        budget = check_token_budget(
            system_prompt="You are helpful.",
            user_prompt="Summarise this: Hello world.",
            model="mistral",
        )
        assert budget.is_safe is True
        assert budget.overflow_by == 0

    def test_total_input_is_sum_of_parts(self):
        system = "System prompt here."
        user   = "User prompt here."
        budget = check_token_budget(system, user, "mistral")

        system_count = count_tokens(system)
        user_count   = count_tokens(user)

        assert budget.system_tokens == system_count
        assert budget.user_tokens   == user_count
        assert budget.total_input_tokens == system_count + user_count

    def test_overflow_detected_on_huge_input(self):
        # Create input that definitely exceeds any model's limit
        huge_text = "word " * 10_000   # ~10,000 tokens
        budget = check_token_budget(
            system_prompt="Be brief.",
            user_prompt=huge_text,
            model="mistral",
        )
        assert budget.is_safe is False
        assert budget.overflow_by > 0

    def test_context_window_matches_model(self):
        budget = check_token_budget("sys.", "usr.", "mistral")
        assert budget.context_window == MODEL_CONTEXT_WINDOWS["mistral"]


class TestTruncateToTokenLimit:
    """Tests for truncate_to_token_limit()"""

    def test_short_text_not_truncated(self):
        text = "Hello world."
        result, was_truncated = truncate_to_token_limit(text, max_tokens=1000)
        assert was_truncated is False
        assert result == text

    def test_long_text_is_truncated(self):
        long_text = "The quick brown fox. " * 500   # ~1500 tokens
        result, was_truncated = truncate_to_token_limit(
            long_text, max_tokens=100
        )
        assert was_truncated is True
        assert count_tokens(result) <= 100 + 5   # small tolerance

    def test_truncated_text_is_shorter(self):
        long_text = "A sentence here. " * 300
        result, _ = truncate_to_token_limit(long_text, max_tokens=50)
        assert len(result) < len(long_text)

    def test_truncation_ends_at_sentence_boundary(self):
        # Text that will definitely be truncated
        text = ("The scientists discovered something important. " * 200)
        result, was_truncated = truncate_to_token_limit(text, max_tokens=50)
        if was_truncated and len(result) > 0:
            # Should end with sentence punctuation
            assert result[-1] in ".!?"


class TestSessionUsage:
    """Tests for SessionUsage accumulation"""

    def test_empty_session_has_zero_totals(self):
        session = SessionUsage()
        assert session.total_tokens == 0
        assert session.file_count   == 0

    def test_add_single_record(self):
        session = SessionUsage()
        session.add(UsageRecord(
            filename="article1.txt",
            prompt_tokens=800,
            completion_tokens=120,
            total_tokens=920,
        ))
        assert session.file_count          == 1
        assert session.total_tokens        == 920
        assert session.total_prompt_tokens == 800

    def test_add_multiple_records_sums_correctly(self):
        session = SessionUsage()
        session.add(UsageRecord("a.txt", 800, 100, 900))
        session.add(UsageRecord("b.txt", 750, 110, 860))

        assert session.file_count          == 2
        assert session.total_prompt_tokens == 1550
        assert session.total_tokens        == 1760

    def test_report_contains_filenames(self):
        session = SessionUsage()
        session.add(UsageRecord("myfile.txt", 800, 100, 900))
        report = session.report()
        assert "myfile.txt" in report

    def test_report_contains_total(self):
        session = SessionUsage()
        session.add(UsageRecord("f.txt", 800, 100, 900))
        report = session.report()
        assert "TOTAL" in report