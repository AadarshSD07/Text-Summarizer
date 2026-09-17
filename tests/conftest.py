"""
conftest.py — Shared pytest fixtures available to all test files.

pytest automatically discovers this file and makes every fixture
defined here available to every test in the tests/ directory.
No import needed — pytest injects them by parameter name.
"""

import pytest
from pathlib import Path
import tempfile
import os


# ---------------------------------------------------------------------------
# Text content fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_article_short():
    """A short but realistic article for testing."""
    return (
        "Scientists at NASA confirmed the discovery of water ice "
        "near the lunar south pole. The finding has significant "
        "implications for future crewed missions to the Moon. "
        "Researchers believe the ice could be used for drinking "
        "water and rocket fuel production."
    )


@pytest.fixture
def sample_article_long():
    """A longer article that exercises token counting."""
    return (
        "The James Webb Space Telescope has fundamentally changed "
        "our understanding of the early universe. Launched in "
        "December 2021, the telescope began sending back data in "
        "2022 that revealed galaxies forming just 300 million years "
        "after the Big Bang. These findings challenge existing models "
        "of galaxy formation. The telescope uses infrared technology "
        "to peer through cosmic dust clouds. Scientists detected "
        "carbon dioxide in an exoplanet atmosphere. "
    ) * 5   # Repeat to make it longer


# ---------------------------------------------------------------------------
# JSON response fixtures — simulate LLM output
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_json_response():
    """A perfectly-formed JSON response from the LLM."""
    return (
        '{\n'
        '  "title": "NASA Confirms Water Ice at Lunar South Pole",\n'
        '  "bullets": [\n'
        '    "NASA scientists confirmed the presence of water ice near '
        'the lunar south pole.",\n'
        '    "The discovery has major implications for future crewed '
        'Moon missions.",\n'
        '    "The ice could be converted into drinking water and rocket '
        'fuel for deep space exploration."\n'
        '  ],\n'
        '  "sentiment": "positive"\n'
        '}'
    )


@pytest.fixture
def fenced_json_response(valid_json_response):
    """Valid JSON wrapped in markdown code fences — common LLM behaviour."""
    return f"```json\n{valid_json_response}\n```"


@pytest.fixture
def preamble_json_response(valid_json_response):
    """Valid JSON preceded by explanatory text — another common LLM habit."""
    return f"Here is the summary you requested:\n\n{valid_json_response}"


@pytest.fixture
def trailing_comma_json():
    """JSON with a trailing comma — invalid JSON but fixable."""
    return (
        '{\n'
        '  "title": "Test Title",\n'
        '  "bullets": [\n'
        '    "First bullet point here.",\n'
        '    "Second bullet point here.",\n'
        '    "Third bullet point here.",\n'   # ← trailing comma
        '  ],\n'
        '  "sentiment": "neutral"\n'
        '}'
    )


@pytest.fixture
def missing_field_json():
    """JSON missing the required sentiment field."""
    return (
        '{\n'
        '  "title": "Test Title",\n'
        '  "bullets": ["One.", "Two.", "Three."]\n'
        '}'
    )


@pytest.fixture
def wrong_bullet_count_json():
    """JSON with only 2 bullets instead of 3."""
    return (
        '{\n'
        '  "title": "Test Title",\n'
        '  "bullets": ["Only one.", "Only two."],\n'
        '  "sentiment": "neutral"\n'
        '}'
    )


# ---------------------------------------------------------------------------
# File system fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_texts_dir(tmp_path, sample_article_short):
    """
    A temporary directory containing real .txt files for testing.

    tmp_path is a built-in pytest fixture that provides a unique
    temporary directory per test — automatically cleaned up after.
    """
    texts_dir = tmp_path / "texts"
    texts_dir.mkdir()

    # Write two sample articles
    (texts_dir / "article1.txt").write_text(
        sample_article_short, encoding="utf-8"
    )
    (texts_dir / "article2.txt").write_text(
        "Researchers developed a new battery technology that could "
        "triple electric vehicle range. The lithium-sulfur design "
        "overcomes historical degradation problems. Commercial "
        "production may begin within five years.",
        encoding="utf-8",
    )

    return texts_dir


@pytest.fixture
def tmp_output_dir(tmp_path):
    """A temporary output directory for saving JSON results."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def empty_texts_dir(tmp_path):
    """A temporary directory with no .txt files — edge case."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    return empty_dir