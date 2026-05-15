"""tests/test_preprocessing.py — minimal smoke tests."""

from src.preprocessing import clean_text, preprocess_for_sentiment


def test_clean_text_strips_html():
    assert clean_text("<p>hello</p>") == "hello"


def test_preprocess_handles_empty():
    assert preprocess_for_sentiment("") == ""


def test_preprocess_strips_urls():
    out = preprocess_for_sentiment("check this https://x.com nice")
    assert "http" not in out
