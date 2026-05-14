"""
src/sentiment.py — Transformer-based sentiment classifier.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest
- RoBERTa-base architecture
- Fine-tuned on 124M tweets
- 3 classes: negative, neutral, positive

NEW in this version: confidence-based "uncertain" bucket.
When the model's top-class confidence is below UNCERTAINTY_THRESHOLD,
the comment is flagged as uncertain rather than forced into pos/neu/neg.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
SCORE_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

# If the model's top-class probability is below this, the result is "uncertain".
# 0.60 is a sensible default — captures genuinely ambiguous cases without
# being overly cautious. Lower it (e.g. 0.5) for more uncertain flags.
UNCERTAINTY_THRESHOLD = 0.60


@dataclass
class SentimentResult:
    label: str           # "positive" | "neutral" | "negative" | "uncertain"
    raw_label: str       # the model's actual prediction before uncertainty mask
    confidence: float    # 0..1, the top-class probability
    score: float         # signed score in [-1, +1], confidence-weighted
    is_uncertain: bool   # True if confidence < UNCERTAINTY_THRESHOLD


class SentimentScorer:
    """Loads once, scores many. Use as a singleton in the Streamlit app."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
    ) -> None:
        # Auto-detect device: GPU on Colab if available, else CPU
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
    def score_batch(self, texts: list[str], batch_size: int = 32) -> list[SentimentResult]:
        """Score a list of texts. Empty/very-short texts return neutral-uncertain."""
        results: list[SentimentResult] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Replace empties so the tokenizer doesn't choke
            safe_batch = [t if t and len(t.strip()) > 0 else " " for t in batch]
            enc = self.tokenizer(
                safe_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)

            for j, pred_idx in enumerate(preds):
                raw_label = LABEL_MAP[int(pred_idx)]
                confidence = float(probs[j][pred_idx])
                is_uncertain = confidence < self.uncertainty_threshold

                # If uncertain, surface that — but keep the raw label for analysis
                final_label = "uncertain" if is_uncertain else raw_label
                signed_score = SCORE_MAP[raw_label] * confidence

                results.append(
                    SentimentResult(
                        label=final_label,
                        raw_label=raw_label,
                        confidence=confidence,
                        score=signed_score,
                        is_uncertain=is_uncertain,
                    )
                )
        return results

    def score_one(self, text: str) -> SentimentResult:
        """Convenience: score a single text."""
        return self.score_batch([text])[0]


def engagement_weighted_score(scores: list[float], likes: list[int]) -> float:
    """
    log(1+likes)-weighted mean of signed sentiment scores.
    Returns a value in [-1, +1]. Comments with more likes count proportionally more.
    """
    import math

    if not scores:
        return 0.0
    weights = [math.log1p(max(0, lk)) for lk in likes]
    total_w = sum(weights)
    if total_w == 0:
        return float(sum(scores) / len(scores))
    weighted = sum(s * w for s, w in zip(scores, weights))
    return float(weighted / total_w)
