"""
src/sentiment.py — Transformer sentiment classifier with uncertainty flag.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
SCORE_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
UNCERTAINTY_THRESHOLD = 0.60


@dataclass
class SentimentResult:
    label: str
    raw_label: str
    confidence: float
    score: float
    is_uncertain: bool


class SentimentScorer:
    def __init__(self, model_name=MODEL_NAME, device=None, uncertainty_threshold=UNCERTAINTY_THRESHOLD):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.uncertainty_threshold = uncertainty_threshold

        print(f"Loading {model_name} on {device}…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        print(f"✓ Model ready on {device}")

    @torch.no_grad()
    def score_batch(self, texts, batch_size=32):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            safe_batch = [t if t and len(t.strip()) > 0 else " " for t in batch]
            enc = self.tokenizer(
                safe_batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)

            for j, pred_idx in enumerate(preds):
                raw_label = LABEL_MAP[int(pred_idx)]
                confidence = float(probs[j][pred_idx])
                is_uncertain = confidence < self.uncertainty_threshold
                final_label = "uncertain" if is_uncertain else raw_label
                signed_score = SCORE_MAP[raw_label] * confidence

                results.append(SentimentResult(
                    label=final_label,
                    raw_label=raw_label,
                    confidence=confidence,
                    score=signed_score,
                    is_uncertain=is_uncertain,
                ))
        return results

    def score_one(self, text):
        return self.score_batch([text])[0]


def engagement_weighted_score(scores, likes):
    import math
    if not scores:
        return 0.0
    weights = [math.log1p(max(0, lk)) for lk in likes]
    total_w = sum(weights)
    if total_w == 0:
        return float(sum(scores) / len(scores))
    weighted = sum(s * w for s, w in zip(scores, weights))
    return float(weighted / total_w)
