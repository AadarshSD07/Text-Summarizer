"""
Tests for summarizer.prompt_builder

Prompts are the interface between your code and the model's behaviour.
These tests verify that prompts contain the required elements and that
the correction prompt properly includes failure information.
"""

import pytest
from summarizer.prompt_builder import (
    build_summary_prompt,
    build_correction_prompt,
    estimate_prompt_length,
    PromptPair,
    FEW_SHOT_EXAMPLE,
)


class TestBuildSummaryPrompt:
    """Tests for build_summary_prompt()"""

    def test_returns_prompt_pair(self, sample_article_short):
        result = build_summary_prompt(sample_article_short, "test.txt")
        assert isinstance(result, PromptPair)

    def test_system_prompt_contains_role(self, sample_article_short):
        pair = build_summary_prompt(sample_article_short, "test.txt")
        # System prompt should define the model's role
        assert "analyst" in pair.system.lower() or "specialist" in pair.system.lower()

    def test_system_prompt_contains_json_requirement(self, sample_article_short):
        pair = build_summary_prompt(sample_article_short, "test.txt")
        assert "JSON" in pair.system

    def test_system_prompt_contains_sentiment_options(self, sample_article_short):
        pair = build_summary_prompt(sample_article_short, "test.txt")
        assert "positive" in pair.system
        assert "negative" in pair.system
        assert "neutral"  in pair.system

    def test_user_prompt_contains_article_text(self, sample_article_short):
        pair = build_summary_prompt(sample_article_short, "test.txt")
        assert sample_article_short in pair.user

    def test_user_prompt_contains_filename(self, sample_article_short):
        pair = build_summary_prompt(sample_article_short, "test.txt")
        assert "test.txt" in pair.user

    def test_include_example_true_adds_example(self, sample_article_short):
        pair = build_summary_prompt(
            sample_article_short, "test.txt", include_example=True
        )
        assert FEW_SHOT_EXAMPLE in pair.system

    def test_include_example_false_omits_example(self, sample_article_short):
        pair = build_summary_prompt(
            sample_article_short, "test.txt", include_example=False
        )
        assert FEW_SHOT_EXAMPLE not in pair.system

    def test_system_prompt_is_non_empty_string(self, sample_article_short):
        pair = build_summary_prompt(sample_article_short)
        assert isinstance(pair.system, str)
        assert len(pair.system) > 0

    def test_user_prompt_is_non_empty_string(self, sample_article_short):
        pair = build_summary_prompt(sample_article_short)
        assert isinstance(pair.user, str)
        assert len(pair.user) > 0


class TestBuildCorrectionPrompt:
    """Tests for build_correction_prompt()"""

    def test_returns_prompt_pair(self, sample_article_short):
        pair = build_correction_prompt(
            original_article=sample_article_short,
            bad_output='{"bad": "json"}',
            error_message="Missing field: bullets",
            error_kind="missing_field",
            attempt_number=1,
        )
        assert isinstance(pair, PromptPair)

    def test_user_prompt_contains_bad_output(self, sample_article_short):
        bad = '{"bad": "json"}'
        pair = build_correction_prompt(
            sample_article_short, bad, "Error msg", "missing_field", 1
        )
        assert bad in pair.user

    def test_user_prompt_contains_error_message(self, sample_article_short):
        error_msg = "Missing required field: bullets"
        pair = build_correction_prompt(
            sample_article_short, "{}", error_msg, "missing_field", 1
        )
        assert error_msg in pair.user

    def test_user_prompt_contains_error_kind(self, sample_article_short):
        pair = build_correction_prompt(
            sample_article_short, "{}", "msg", "syntax_error", 1
        )
        assert "syntax_error" in pair.user

    def test_user_prompt_contains_original_article(self, sample_article_short):
        pair = build_correction_prompt(
            sample_article_short, "{}", "msg", "missing_field", 1
        )
        assert sample_article_short in pair.user

    def test_attempt_number_included(self, sample_article_short):
        pair = build_correction_prompt(
            sample_article_short, "{}", "msg", "missing_field", 2
        )
        assert "2" in pair.user


class TestEstimatePromptLength:
    """Tests for estimate_prompt_length()"""

    def test_returns_dict_with_required_keys(self, sample_article_short):
        pair = build_summary_prompt(sample_article_short)
        lengths = estimate_prompt_length(pair)

        assert "system_chars"  in lengths
        assert "user_chars"    in lengths
        assert "total_chars"   in lengths
        assert "approx_tokens" in lengths

    def test_total_is_sum_of_parts(self, sample_article_short):
        pair    = build_summary_prompt(sample_article_short)
        lengths = estimate_prompt_length(pair)

        assert lengths["total_chars"] == (
            lengths["system_chars"] + lengths["user_chars"]
        )

    def test_approx_tokens_positive(self, sample_article_short):
        pair    = build_summary_prompt(sample_article_short)
        lengths = estimate_prompt_length(pair)
        assert lengths["approx_tokens"] > 0