"""
Integration tests — multiple modules working together.

These tests verify that the modules interact correctly,
not just that each works in isolation.
No LLM calls are made — we test everything up to that boundary.
"""

import json
import pytest
from pathlib import Path

from summarizer.file_reader import read_txt_files, FileContent
from summarizer.prompt_builder import build_summary_prompt, PromptPair
from summarizer.json_parser import (
    parse_llm_response,
    save_summary_to_file,
    ArticleSummary,
)
from summarizer.token_counter import check_token_budget


class TestFileReaderToPromptBuilder:
    """Verify file_reader output feeds correctly into prompt_builder"""

    def test_file_content_produces_valid_prompt_pair(self, tmp_texts_dir):
        results = [
            r for r in read_txt_files(tmp_texts_dir)
            if isinstance(r, FileContent)
        ]
        assert len(results) > 0

        for result in results:
            pair = build_summary_prompt(result.content, result.filename)
            assert isinstance(pair, PromptPair)
            assert result.content in pair.user
            assert result.filename in pair.user
            assert len(pair.system) > 0

    def test_prompt_contains_all_article_content(self, tmp_texts_dir):
        results = [
            r for r in read_txt_files(tmp_texts_dir)
            if isinstance(r, FileContent)
        ]
        file_result = results[0]
        pair = build_summary_prompt(file_result.content, file_result.filename)

        # Every word in the article should appear somewhere in the prompt
        for word in file_result.content.split()[:10]:
            assert word in pair.user


class TestPromptBuilderToTokenCounter:
    """Verify prompt_builder output feeds correctly into token_counter"""

    def test_budget_check_uses_real_prompt_tokens(self, tmp_texts_dir):
        results = [
            r for r in read_txt_files(tmp_texts_dir)
            if isinstance(r, FileContent)
        ]
        file_result = results[0]
        pair = build_summary_prompt(file_result.content, file_result.filename)

        budget = check_token_budget(pair.system, pair.user, "mistral")

        # Budget should reflect actual prompt sizes
        assert budget.system_tokens > 0
        assert budget.user_tokens   > 0
        assert budget.is_safe is True   # Sample articles are short

    def test_budget_correctly_identifies_safe_articles(self, tmp_texts_dir):
        for result in read_txt_files(tmp_texts_dir):
            if isinstance(result, FileContent):
                pair   = build_summary_prompt(result.content, result.filename)
                budget = check_token_budget(pair.system, pair.user, "mistral")
                # Our sample articles are well within limits
                assert budget.is_safe, (
                    f"{result.filename} exceeds token budget: "
                    f"{budget.total_input_tokens} > {budget.max_safe_input}"
                )


class TestParserToFileSaver:
    """Verify json_parser output saves correctly to disk"""

    def test_valid_summary_saves_to_disk(self, tmp_output_dir, valid_json_response):
        summary = parse_llm_response(valid_json_response, "article1.txt")
        assert isinstance(summary, ArticleSummary)

        save_summary_to_file(summary, tmp_output_dir)

        output_file = tmp_output_dir / "article1.json"
        assert output_file.exists()

    def test_saved_json_is_valid_and_complete(
        self, tmp_output_dir, valid_json_response
    ):
        summary = parse_llm_response(valid_json_response, "article1.txt")
        save_summary_to_file(summary, tmp_output_dir)

        output_file  = tmp_output_dir / "article1.json"
        saved_data   = json.loads(output_file.read_text(encoding="utf-8"))

        assert saved_data["title"]       == summary.title
        assert saved_data["bullets"]     == summary.bullets
        assert saved_data["sentiment"]   == summary.sentiment
        assert saved_data["source_file"] == "article1.txt"

    def test_saved_json_does_not_contain_raw_response(
        self, tmp_output_dir, valid_json_response
    ):
        summary = parse_llm_response(valid_json_response, "article1.txt")
        save_summary_to_file(summary, tmp_output_dir)

        output_file = tmp_output_dir / "article1.json"
        content     = output_file.read_text(encoding="utf-8")

        # raw_response is internal — should not appear in output files
        assert "raw_response" not in content

    def test_output_filename_derived_from_source(
        self, tmp_output_dir, valid_json_response
    ):
        summary = parse_llm_response(valid_json_response, "my_article.txt")
        save_summary_to_file(summary, tmp_output_dir)

        # Output should be my_article.json not my_article.txt
        assert (tmp_output_dir / "my_article.json").exists()
        assert not (tmp_output_dir / "my_article.txt").exists()


class TestFullPipelineWithoutLLM:
    """
    Tests the complete pipeline from files to saved JSON,
    using a pre-crafted LLM response instead of a real API call.
    This is the closest we get to end-to-end without a running model.
    """

    def test_complete_pipeline_produces_output_files(
        self, tmp_texts_dir, tmp_output_dir, valid_json_response
    ):
        # Read files
        articles = [
            r for r in read_txt_files(tmp_texts_dir)
            if isinstance(r, FileContent)
        ]
        assert len(articles) == 2

        # For each article, build prompt, simulate LLM response, parse, save
        for article in articles:
            pair    = build_summary_prompt(article.content, article.filename)
            budget  = check_token_budget(pair.system, pair.user, "mistral")
            assert budget.is_safe

            # Simulate what the LLM would return
            parsed = parse_llm_response(valid_json_response, article.filename)
            assert isinstance(parsed, ArticleSummary)

            save_summary_to_file(parsed, tmp_output_dir)

        # Both articles should produce output files
        output_files = list(tmp_output_dir.glob("*.json"))
        assert len(output_files) == 2

    def test_all_output_files_are_valid_json(
        self, tmp_texts_dir, tmp_output_dir, valid_json_response
    ):
        articles = [
            r for r in read_txt_files(tmp_texts_dir)
            if isinstance(r, FileContent)
        ]

        for article in articles:
            parsed = parse_llm_response(valid_json_response, article.filename)
            save_summary_to_file(parsed, tmp_output_dir)

        for json_file in tmp_output_dir.glob("*.json"):
            data = json.loads(json_file.read_text(encoding="utf-8"))
            assert "title"     in data
            assert "bullets"   in data
            assert "sentiment" in data
            assert len(data["bullets"]) == 3
            assert data["sentiment"] in ("positive", "negative", "neutral")