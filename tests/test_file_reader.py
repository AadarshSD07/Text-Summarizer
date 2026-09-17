"""
Tests for summarizer.file_reader

Strategy: use real files in temporary directories (tmp_path fixture).
We test the happy path, encoding fallback, empty files, and non-txt files.
"""

import pytest
from pathlib import Path
from summarizer.file_reader import (
    discover_txt_files,
    read_single_file,
    read_txt_files,
    FileContent,
    FileError,
    summarize_read_results,
)


class TestDiscoverTxtFiles:
    """Tests for discover_txt_files()"""

    def test_finds_txt_files(self, tmp_texts_dir):
        files = discover_txt_files(tmp_texts_dir)
        assert len(files) == 2

    def test_returns_sorted_list(self, tmp_texts_dir):
        files = discover_txt_files(tmp_texts_dir)
        names = [f.name for f in files]
        assert names == sorted(names)

    def test_ignores_non_txt_files(self, tmp_path):
        folder = tmp_path / "mixed"
        folder.mkdir()
        (folder / "article.txt").write_text("content", encoding="utf-8")
        (folder / "notes.pdf").write_bytes(b"%PDF-1.4")
        (folder / "data.csv").write_text("a,b,c", encoding="utf-8")

        files = discover_txt_files(folder)

        # Only the .txt file should be found
        assert len(files) == 1
        assert files[0].name == "article.txt"

    def test_empty_directory_returns_empty_list(self, empty_texts_dir):
        files = discover_txt_files(empty_texts_dir)
        assert files == []

    def test_nonexistent_directory_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            discover_txt_files(missing)

    def test_file_path_raises_not_directory(self, tmp_path):
        a_file = tmp_path / "not_a_dir.txt"
        a_file.write_text("content", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            discover_txt_files(a_file)


class TestReadSingleFile:
    """Tests for read_single_file()"""

    def test_reads_utf8_file(self, tmp_path, sample_article_short):
        f = tmp_path / "test.txt"
        f.write_text(sample_article_short, encoding="utf-8")

        result = read_single_file(f)

        assert isinstance(result, FileContent)
        assert result.content == sample_article_short.strip()
        assert result.encoding_used == "utf-8"
        assert result.filename == "test.txt"
        assert result.name == "test"

    def test_reads_latin1_file(self, tmp_path):
        # Write a file with latin-1 encoding (Windows legacy)
        f = tmp_path / "legacy.txt"
        f.write_bytes("Caf\xe9 au lait is delicious.".encode("latin-1"))

        result = read_single_file(f)

        assert isinstance(result, FileContent)
        assert result.encoding_used == "latin-1"
        assert "Caf" in result.content

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "padded.txt"
        f.write_text("\n\n  Content here.  \n\n", encoding="utf-8")

        result = read_single_file(f)

        assert isinstance(result, FileContent)
        assert result.content == "Content here."

    def test_char_count_correct(self, tmp_path):
        text = "Exactly twenty chars."
        f = tmp_path / "counted.txt"
        f.write_text(text, encoding="utf-8")

        result = read_single_file(f)

        assert isinstance(result, FileContent)
        assert result.char_count == len(text)

    def test_empty_file_returns_file_content_with_zero_chars(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")

        result = read_single_file(f)

        # Empty file is readable — just has no content
        assert isinstance(result, FileContent)
        assert result.char_count == 0
        assert result.content == ""


class TestReadTxtFiles:
    """Tests for the read_txt_files() generator"""

    def test_yields_file_content_for_each_file(self, tmp_texts_dir):
        results = list(read_txt_files(tmp_texts_dir))

        assert len(results) == 2
        assert all(isinstance(r, FileContent) for r in results)

    def test_max_files_limits_results(self, tmp_texts_dir):
        results = list(read_txt_files(tmp_texts_dir, max_files=1))
        assert len(results) == 1

    def test_empty_directory_yields_nothing(self, empty_texts_dir):
        results = list(read_txt_files(empty_texts_dir))
        assert results == []

    def test_is_generator(self, tmp_texts_dir):
        import types
        result = read_txt_files(tmp_texts_dir)
        assert isinstance(result, types.GeneratorType)


class TestSummarizeReadResults:
    """Tests for summarize_read_results()"""

    def test_counts_successes_and_failures(self, tmp_texts_dir):
        results = list(read_txt_files(tmp_texts_dir))
        summary = summarize_read_results(results)

        assert summary["total"] == 2
        assert summary["succeeded"] == 2
        assert summary["failed"] == 0

    def test_total_chars_sums_correctly(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("Hello.", encoding="utf-8")
        f2.write_text("World.", encoding="utf-8")

        results = list(read_txt_files(tmp_path))
        summary = summarize_read_results(results)

        assert summary["total_chars"] == len("Hello.") + len("World.")