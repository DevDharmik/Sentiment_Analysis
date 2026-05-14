"""
src/preprocessing.py — Light text cleaning for YouTube comments.

Designed for transformers + classical models. Removes noise (HTML, URLs),
preserves signal (emojis converted to text tokens), and detects language.
"""

from __future__ import annotations

import re
from html import unescape

import demoji
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 42  # deterministic language detection

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def clean_text(text: str) -> str:
    """Strip HTML tags, decode entities, normalise whitespace."""
    if not text:
        return ""
    text = unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def emoji_to_text(text: str) -> str:
    """Convert 🔥 → :fire: so the tokenizer can read emojis as words."""
    if not text:
        return ""
    return demoji.replace_with_desc(text, sep=":")


def strip_urls(text: str) -> str:
    """Remove URLs — usually spam, rarely sentiment-bearing."""
    return _URL_RE.sub("", text).strip()


def detect_language(text: str, default: str = "unknown") -> str:
    """Return ISO-639-1 code ('en', 'es', 'hi'...) or 'unknown' for short/empty input."""
    if not text or len(text.strip()) < 5:
        return default
    try:
        return detect(text)
    except LangDetectException:
        return default


def preprocess_for_sentiment(text: str) -> str:
    """
    Full pipeline applied before sending text to the sentiment model.
    Order matters: HTML first, then URLs (so URL inside HTML gets caught),
    then emojis, then whitespace normalisation.
    """
    text = clean_text(text)
    text = strip_urls(text)
    text = emoji_to_text(text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text
