"""
src/explain.py — Token-level attribution for linear classical models.


For a linear classifier (LinearSVC, LogisticRegression), each prediction
can be decomposed exactly into per-token contributions:
    contribution(token) = tfidf_value(token) × coefficient(token)

Fast, exact, dep-free alternative to shap.LinearExplainer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TokenContribution:
    token: str
    tfidf: float
    coefficient: float
    contribution: float  # tfidf * coef; + = positive class, - = negative


def explain_linear(model, vectorizer, text: str, top_k: int = 15) -> list:
    """
    Return the top-k tokens (by absolute contribution) for a single text.
    Works on any sklearn linear binary classifier exposing `.coef_`.
    """
    if not hasattr(model, "coef_"):
        raise TypeError(f"{type(model).__name__} has no .coef_ — not a linear model")

    coef = model.coef_.ravel()
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    contribs = []
    _, cols = X.nonzero()
    for col in cols:
        tfidf = float(X[0, col])
        c = float(coef[col])
        contribs.append(TokenContribution(
            token=feature_names[col],
            tfidf=tfidf,
            coefficient=c,
            contribution=tfidf * c,
        ))

    contribs.sort(key=lambda x: abs(x.contribution), reverse=True)
    return contribs[:top_k]


def predicted_class(model, vectorizer, text: str) -> Tuple[str, float]:
    """Return (label, decision_score). Higher absolute decision = more confident."""
    X = vectorizer.transform([text])
    label = model.predict(X)[0]
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
    elif hasattr(model, "predict_proba"):
        score = float(max(model.predict_proba(X)[0]))
    else:
        score = 0.0
    return label, score


def explain_text_html(model, vectorizer, text: str, top_k: int = 15) -> str:
    """
    Generate HTML showing the text with tokens colored by contribution.
    Green = pushes toward positive class, red = pushes toward negative.
    Intensity scales with absolute contribution magnitude.
    """
    contribs = explain_linear(model, vectorizer, text, top_k=top_k)
    token_map = {c.token.lower(): c.contribution for c in contribs}

    tokens = re.findall(r'\S+|\s+', text)
    parts = []
    for tok in tokens:
        clean = tok.lower().strip('.,!?;:"\'()[]')
        contrib = token_map.get(clean, 0.0)
        if contrib != 0.0:
            intensity = min(abs(contrib) / 0.5, 1.0)
            if contrib > 0:
                color = f"rgba(34, 197, 94, {intensity * 0.7 + 0.2})"
            else:
                color = f"rgba(239, 68, 68, {intensity * 0.7 + 0.2})"
            parts.append(
                f'<span style="background-color: {color}; padding: 2px 4px; '
                f'border-radius: 3px; color: white;" '
                f'title="contribution: {contrib:+.3f}">{tok}</span>'
            )
        else:
            parts.append(tok)
    return ''.join(parts)
