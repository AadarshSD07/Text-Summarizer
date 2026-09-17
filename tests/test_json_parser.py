"""
Tests for summarizer.json_parser

This module has the most edge cases in the entire project — it's
responsible for turning unpredictable LLM output into reliable data.
Therefore it gets the most thorough test coverage.
"""

import pytest
from summarizer.json_parser import (
    clean_llm_output,
    fix_common_json_errors,
    parse_llm_response,
    validate_summary_schema,
    ArticleSummary,
    ParseError,
    ErrorKind,
)


class TestCleanLlmOutput:
    """Tests for the cleaning/extraction stage"""

    def test_plain_json_unchanged(self, valid_json_response):
        result = clean_llm_output(valid_json_response)
        assert result is not None
        assert result.startswith("{")
        assert result.endswith("}")

    def test_strips_json_code_fence(self, fenced_json_response):
        result = clean_llm_output(fenced_json_response)
        assert result is not None
        assert "```" not in result

    def test_strips_plain_code_fence(self, valid_json_response):
        fenced = f"```\n{valid_json_response}\n```"
        result = clean_llm_output(fenced)
        assert result is not None
        assert "```" not in result

    def test_strips_preamble(self, preamble_json_response):
        result = clean_llm_output(preamble_json_response)
        assert result is not None
        assert result.startswith("{")

    def test_strips_postamble(self, valid_json_response):
        with_postamble = f"{valid_json_response}\n\nLet me know if you need changes!"
        result = clean_llm_output(with_postamble)
        assert result is not None
        assert "Let me know" not in result

    def test_returns_none_for_empty_string(self):
        assert clean_llm_output("") is None

    def test_returns_none_for_whitespace_only(self):
        assert clean_llm_output("   \n  ") is None

    def test_returns_none_for_no_json(self):
        assert clean_llm_output("This has no JSON at all.") is None

    def test_returns_none_for_truncated_json(self):
        truncated = '{"title": "Incomplete'
        assert clean_llm_output(truncated) is None

    def test_handles_nested_objects(self):
        nested = '{"outer": {"inner": "value"}, "key": "val"}'
        result = clean_llm_output(nested)
        assert result is not None
        assert '"inner"' in result


class TestFixCommonJsonErrors:
    """Tests for the JSON repair stage"""

    def test_fixes_trailing_comma_in_array(self, trailing_comma_json):
        fixed = fix_common_json_errors(trailing_comma_json)
        import json
        # Should now be parseable
        data = json.loads(fixed)
        assert len(data["bullets"]) == 3

    def test_fixes_trailing_comma_in_object(self):
        text = '{"key": "value",}'
        fixed = fix_common_json_errors(text)
        import json
        data = json.loads(fixed)
        assert data["key"] == "value"

    def test_lowercases_sentiment(self):
        text = '{"sentiment": "Positive"}'
        fixed = fix_common_json_errors(text)
        assert '"positive"' in fixed

    def test_lowercases_sentiment_all_caps(self):
        text = '{"sentiment": "NEGATIVE"}'
        fixed = fix_common_json_errors(text)
        assert '"negative"' in fixed

    def test_does_not_alter_valid_json(self, valid_json_response):
        fixed = fix_common_json_errors(valid_json_response)
        import json
        # Should still parse correctly
        data = json.loads(fixed)
        assert "title" in data


class TestValidateSummarySchema:
    """Tests for the schema validation stage"""

    def test_valid_data_returns_none(self):
        data = {
            "title": "Test Title Here",
            "bullets": ["One.", "Two.", "Three."],
            "sentiment": "positive",
        }
        assert validate_summary_schema(data) is None

    def test_missing_title_returns_error(self):
        data = {"bullets": ["a.", "b.", "c."], "sentiment": "neutral"}
        error = validate_summary_schema(data)
        assert isinstance(error, ParseError)
        assert error.error_kind == ErrorKind.MISSING_FIELD
        assert "title" in error.field_name

    def test_missing_bullets_returns_error(self):
        data = {"title": "T", "sentiment": "neutral"}
        error = validate_summary_schema(data)
        assert isinstance(error, ParseError)
        assert error.error_kind == ErrorKind.MISSING_FIELD

    def test_missing_sentiment_returns_error(self):
        data = {"title": "T", "bullets": ["a.", "b.", "c."]}
        error = validate_summary_schema(data)
        assert isinstance(error, ParseError)
        assert error.error_kind == ErrorKind.MISSING_FIELD

    def test_wrong_bullet_count_low(self):
        data = {
            "title": "T",
            "bullets": ["One.", "Two."],      # Only 2
            "sentiment": "neutral",
        }
        error = validate_summary_schema(data)
        assert isinstance(error, ParseError)
        assert error.error_kind == ErrorKind.WRONG_LENGTH

    def test_wrong_bullet_count_high(self):
        data = {
            "title": "T",
            "bullets": ["1.", "2.", "3.", "4."],   # 4 instead of 3
            "sentiment": "neutral",
        }
        error = validate_summary_schema(data)
        assert isinstance(error, ParseError)
        assert error.error_kind == ErrorKind.WRONG_LENGTH

    def test_invalid_sentiment_value(self):
        data = {
            "title": "T",
            "bullets": ["a.", "b.", "c."],
            "sentiment": "mixed",             # Not in allowed set
        }
        error = validate_summary_schema(data)
        assert isinstance(error, ParseError)
        assert error.error_kind == ErrorKind.WRONG_VALUE

    def test_bullets_as_string_not_list(self):
        data = {
            "title": "T",
            "bullets": "This should be a list.",
            "sentiment": "neutral",
        }
        error = validate_summary_schema(data)
        assert isinstance(error, ParseError)
        assert error.error_kind == ErrorKind.WRONG_TYPE

    def test_all_valid_sentiments_accepted(self):
        for sentiment in ("positive", "negative", "neutral"):
            data = {
                "title": "T",
                "bullets": ["a.", "b.", "c."],
                "sentiment": sentiment,
            }
            assert validate_summary_schema(data) is None


class TestParseLlmResponse:
    """End-to-end tests for the full parse pipeline"""

    def test_parses_valid_json(self, valid_json_response):
        result = parse_llm_response(valid_json_response, "test.txt")
        assert isinstance(result, ArticleSummary)
        assert result.source_file == "test.txt"
        assert len(result.bullets) == 3
        assert result.sentiment in ("positive", "negative", "neutral")

    def test_parses_fenced_json(self, fenced_json_response):
        result = parse_llm_response(fenced_json_response, "test.txt")
        assert isinstance(result, ArticleSummary)

    def test_parses_json_with_preamble(self, preamble_json_response):
        result = parse_llm_response(preamble_json_response, "test.txt")
        assert isinstance(result, ArticleSummary)

    def test_parses_json_with_trailing_comma(self, trailing_comma_json):
        result = parse_llm_response(trailing_comma_json, "test.txt")
        assert isinstance(result, ArticleSummary)

    def test_empty_input_returns_error(self):
        result = parse_llm_response("", "test.txt")
        assert isinstance(result, ParseError)
        assert result.error_kind == ErrorKind.EMPTY_CONTENT

    def test_no_json_returns_error(self):
        result = parse_llm_response("No JSON here at all.", "test.txt")
        assert isinstance(result, ParseError)
        assert result.error_kind == ErrorKind.NO_JSON

    def test_missing_field_returns_error(self, missing_field_json):
        result = parse_llm_response(missing_field_json, "test.txt")
        assert isinstance(result, ParseError)
        assert result.error_kind == ErrorKind.MISSING_FIELD

    def test_wrong_bullet_count_returns_error(self, wrong_bullet_count_json):
        result = parse_llm_response(wrong_bullet_count_json, "test.txt")
        assert isinstance(result, ParseError)
        assert result.error_kind == ErrorKind.WRONG_LENGTH

    def test_result_stores_source_file(self, valid_json_response):
        result = parse_llm_response(valid_json_response, "article1.txt")
        assert isinstance(result, ArticleSummary)
        assert result.source_file == "article1.txt"

    def test_sentiment_is_normalised_to_lowercase(self):
        raw = (
            '{"title": "T", '
            '"bullets": ["a.", "b.", "c."], '
            '"sentiment": "POSITIVE"}'
        )
        result = parse_llm_response(raw, "test.txt")
        assert isinstance(result, ArticleSummary)
        assert result.sentiment == "positive"

    def test_raw_response_preserved_on_error(self):
        raw = "This is not JSON."
        result = parse_llm_response(raw, "test.txt")
        assert isinstance(result, ParseError)
        assert result.raw_response == raw

    @pytest.mark.parametrize("sentiment", ["positive", "negative", "neutral"])
    def test_all_valid_sentiments(self, sentiment):
        raw = (
            f'{{"title": "T", '
            f'"bullets": ["a.", "b.", "c."], '
            f'"sentiment": "{sentiment}"}}'
        )
        result = parse_llm_response(raw)
        assert isinstance(result, ArticleSummary)
        assert result.sentiment == sentiment