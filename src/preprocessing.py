"""
src/preprocessing.py — Light text cleaning for YouTube comments.
"""


from __future__ import annotations

import re
from html import unescape

import demoji
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 42

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def emoji_to_text(text: str) -> str:
    if not text:
        return ""
    return demoji.replace_with_desc(text, sep=":")


def strip_urls(text: str) -> str:
    return _URL_RE.sub("", text).strip()


def detect_language(text: str, default: str = "unknown") -> str:
    if not text or len(text.strip()) < 5:
        return default
    try:
        return detect(text)
    except LangDetectException:
        return default


def preprocess_for_sentiment(text: str) -> str:
    text = clean_text(text)
    text = strip_urls(text)
    text = emoji_to_text(text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text
