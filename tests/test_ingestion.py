"""Loading and v1 cleaning (kept so the A0 baseline stays covered), plus manifest."""

from src.ingestion.loader import load_all_transcripts, load_manifest, parse_filename
from src.ingestion.parser import clean_transcript, extract_paragraphs, get_text_stats


class TestParseFilename:
    def test_valid_filename(self):
        r = parse_filename("NVDA_Q1_2024.txt")
        assert (r["ticker"], r["quarter"], r["year"], r["company"]) == ("NVDA", "Q1", 2024, "NVIDIA")

    def test_valid_filename_msft(self):
        r = parse_filename("MSFT_Q3_2024.txt")
        assert (r["ticker"], r["quarter"], r["year"], r["company"]) == ("MSFT", "Q3", 2024, "Microsoft")

    def test_unknown_ticker_falls_back(self):
        assert parse_filename("XYZ_Q2_2023.txt")["company"] == "XYZ"

    def test_invalid_filename_returns_none(self):
        assert parse_filename("random_file.txt") is None
        assert parse_filename("nvda_q1_2024.txt") is None
        assert parse_filename("NVDA_2024.txt") is None

    def test_all_quarters(self):
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            assert parse_filename(f"AAPL_{q}_2024.txt")["quarter"] == q


class TestCleanTranscriptV1:
    def test_removes_operator_lines(self):
        cleaned = clean_transcript("Operator: Good afternoon.\nWe had strong revenue growth of 40 percent.")
        assert "Good afternoon" not in cleaned and "strong revenue" in cleaned

    def test_removes_timestamps(self):
        cleaned = clean_transcript("[00:12:34] Revenue grew significantly this quarter.")
        assert "[00:12:34]" not in cleaned and "Revenue grew" in cleaned

    def test_empty_input(self):
        assert clean_transcript("") == ""

    def test_reduces_length(self):
        raw = "Operator: Good afternoon. Thank you. Please go ahead.\n" * 10
        raw += "Revenue was a record this quarter driven by strong demand."
        assert len(clean_transcript(raw)) < len(raw)

    def test_stats_reporting(self):
        raw = "Operator: Good afternoon. " * 5 + "Revenue grew 40 percent."
        stats = get_text_stats(raw, clean_transcript(raw))
        assert stats["raw_chars"] == len(raw)


class TestExtractParagraphs:
    def test_splits_on_double_newline(self):
        text = "First paragraph with enough content here.\n\nSecond paragraph with enough content here."
        assert len(extract_paragraphs(text)) == 2

    def test_filters_short_paragraphs(self):
        text = "Hi\n\nThis is a real paragraph with enough content to keep."
        assert len(extract_paragraphs(text)) == 1


class TestManifest:
    def test_manifest_fills_call_date_and_url(self, tmp_path):
        (tmp_path / "t").mkdir()
        (tmp_path / "t" / "NVDA_Q4_2026.txt").write_text("Operator: hello there " * 20)
        m = tmp_path / "manifest.csv"
        m.write_text("ticker,fiscal_quarter,fiscal_year,call_date,source_url\n"
                     "NVDA,Q4,2026,2026-02-25,https://example.com/x\n")
        assert load_manifest(m)[("NVDA", "Q4", 2026)]["call_date"] == "2026-02-25"
        t = load_all_transcripts(tmp_path / "t", m)[0]
        assert t["date"] == "2026-02-25" and t["source_url"] == "https://example.com/x"

    def test_missing_manifest_is_tolerated(self, tmp_path):
        assert load_manifest(tmp_path / "nope.csv") == {}
