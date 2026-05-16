"""
YouTube Comment Sentiment Analysis Dashboard
Uses YouTube Data API v3 + HuggingFace RoBERTa as the main classifier.
Compares against the 3 classical models from src/train_classical.py
(SVM, Logistic Regression, Naive Bayes — all sharing one TF-IDF vectorizer).
"""
import os
from urllib.parse import urlparse, parse_qs

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# ---------- Config ----------
st.set_page_config(page_title="YouTube Sentiment Analysis", page_icon="🎬", layout="wide")

API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

MODELS_DIR = "models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")
CLASSICAL_MODELS = {
    "SVM":        os.path.join(MODELS_DIR, "svm_tfidf.joblib"),
    "LogReg":     os.path.join(MODELS_DIR, "logreg_tfidf.joblib"),
    "NaiveBayes": os.path.join(MODELS_DIR, "nb_tfidf.joblib"),
}

ROBERTA_LABEL_MAP = {
    "label_0": "negative", "label_1": "neutral", "label_2": "positive",
    "negative": "negative", "neutral": "neutral", "positive": "positive",
}
CLASSICAL_LABEL_MAP = {"pos": "positive", "neg": "negative"}
COLOR_MAP = {"positive": "#10b981", "neutral": "#94a3b8", "negative": "#ef4444"}


# ---------- Helpers ----------
def extract_video_id(url: str):
    parsed = urlparse(url.strip())
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    if parsed.hostname and "youtube.com" in parsed.hostname:
        return parse_qs(parsed.query).get("v", [None])[0]
    return None


@st.cache_resource(show_spinner=False)
def get_youtube_client():
    return build("youtube", "v3", developerKey=API_KEY)


@st.cache_resource(show_spinner=False)
def get_roberta_pipeline():
    if not HAS_TRANSFORMERS:
        return None
    return pipeline("sentiment-analysis", model=ROBERTA_MODEL, truncation=True, max_length=512)


@st.cache_resource(show_spinner=False)
def get_vectorizer():
    if not HAS_JOBLIB or not os.path.exists(VECTORIZER_PATH):
        return None
    try:
        return joblib.load(VECTORIZER_PATH)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_classical_models():
    if not HAS_JOBLIB:
        return {}
    loaded = {}
    for name, path in CLASSICAL_MODELS.items():
        if os.path.exists(path):
            try:
                loaded[name] = joblib.load(path)
            except Exception:
                pass
    return loaded


def fetch_video_meta(youtube, video_id):
    resp = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    items = resp.get("items", [])
    if not items:
        return None
    s, st_ = items[0]["snippet"], items[0]["statistics"]
    return {
        "title": s["title"],
        "channel": s["channelTitle"],
        "published": s["publishedAt"],
        "thumbnail": s["thumbnails"]["high"]["url"],
        "views": int(st_.get("viewCount", 0)),
        "likes": int(st_.get("likeCount", 0)),
        "comments": int(st_.get("commentCount", 0)),
    }


def fetch_comments(youtube, video_id, max_comments=100):
    comments, next_token = [], None
    while len(comments) < max_comments:
        try:
            resp = youtube.commentThreads().list(
                part="snippet", videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                order="relevance", pageToken=next_token, textFormat="plainText",
            ).execute()
        except HttpError as e:
            st.error(f"YouTube API error: {e}")
            break
        for item in resp.get("items", []):
            snip = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": snip.get("authorDisplayName", ""),
                "text": snip.get("textDisplay", ""),
                "likes": int(snip.get("likeCount", 0)),
                "published": snip.get("publishedAt", ""),
            })
        next_token = resp.get("nextPageToken")
        if not next_token:
            break
    return comments[:max_comments]


def score_roberta(texts, pipe):
    if not pipe or not texts:
        return []
    results = pipe([t[:512] for t in texts], batch_size=16)
    return [ROBERTA_LABEL_MAP.get(r["label"].lower(), r["label"].lower()) for r in results]


def score_classical(texts, model, vectorizer):
    """Transform texts with shared TF-IDF vectorizer, then predict."""
    if not model or vectorizer is None or not texts:
        return []
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    return [CLASSICAL_LABEL_MAP.get(str(p).lower(), str(p).lower()) for p in preds]


# ---------- UI ----------
st.title("🎬 YouTube Comment Sentiment Analysis")
st.caption("Paste any YouTube URL → get sentiment breakdown in seconds.")

if not API_KEY:
    st.error("`YOUTUBE_API_KEY` env var is not set. Add it to your shell or as a Codespaces secret.")
    st.stop()

c1, c2 = st.columns([3, 1])
with c1:
    url = st.text_input("YouTube URL", placeholder="https://youtu.be/...")
with c2:
    max_comments = st.number_input("Max comments", min_value=10, max_value=500, value=100, step=10)

go = st.button("Analyse", type="primary", use_container_width=True)

if go and url:
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Could not parse video ID from URL.")
        st.stop()

    youtube = get_youtube_client()

    with st.spinner("Fetching video metadata..."):
        meta = fetch_video_meta(youtube, video_id)
    if not meta:
        st.error("Video not found or private.")
        st.stop()

    h1, h2 = st.columns([1, 2])
    with h1:
        st.image(meta["thumbnail"], use_container_width=True)
    with h2:
        st.subheader(meta["title"])
        st.write(f"**Channel:** {meta['channel']}  |  **Published:** {meta['published'][:10]}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Views", f"{meta['views']:,}")
        m2.metric("Likes", f"{meta['likes']:,}")
        m3.metric("Comments", f"{meta['comments']:,}")

    with st.spinner(f"Fetching up to {max_comments} comments..."):
        comments = fetch_comments(youtube, video_id, max_comments)
    if not comments:
        st.warning("No comments fetched (comments might be disabled).")
        st.stop()
    st.success(f"Fetched {len(comments)} comments.")

    with st.spinner("Loading RoBERTa (first run downloads ~500 MB)..."):
        pipe = get_roberta_pipeline()
    if not pipe:
        st.error("Could not load RoBERTa.")
        st.stop()

    texts = [c["text"] for c in comments]
    with st.spinner("Scoring with RoBERTa..."):
        roberta_labels = score_roberta(texts, pipe)

    df = pd.DataFrame(comments)
    df["sentiment"] = roberta_labels
    df["published"] = pd.to_datetime(df["published"], errors="coerce")

    vectorizer = get_vectorizer()
    classical = get_classical_models()
    classical_results = {}
    if vectorizer is not None and classical:
        with st.spinner(f"Scoring with {len(classical)} classical models..."):
            for name, model in classical.items():
                try:
                    classical_results[name] = score_classical(texts, model, vectorizer)
                    df[f"sentiment_{name}"] = classical_results[name]
                except Exception as e:
                    st.warning(f"{name} scoring failed: {e}")

    st.divider()
    st.subheader("📊 Sentiment Breakdown (RoBERTa)")

    counts = df["sentiment"].value_counts()
    total = counts.sum()
    pos = int(counts.get("positive", 0))
    neg = int(counts.get("negative", 0))
    neu = int(counts.get("neutral", 0))

    score_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["score"] = df["sentiment"].map(score_map)
    df["weight"] = np.log1p(df["likes"])
    engagement_score = (
        (df["score"] * df["weight"]).sum() / df["weight"].sum()
        if df["weight"].sum() > 0 else df["score"].mean()
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Positive", f"{pos}  ({pos/total*100:.1f}%)")
    k2.metric("Neutral",  f"{neu}  ({neu/total*100:.1f}%)")
    k3.metric("Negative", f"{neg}  ({neg/total*100:.1f}%)")
    k4.metric("Engagement-weighted score", f"{engagement_score:+.3f}",
              help="−1 to +1, weighted by log(1+likes)")

    g1, g2 = st.columns(2)
    with g1:
        fig = px.pie(names=counts.index, values=counts.values, hole=0.5,
                     color=counts.index, color_discrete_map=COLOR_MAP,
                     title="Sentiment distribution (RoBERTa)")
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        if classical_results:
            all_models = {"RoBERTa": df["sentiment"].tolist()}
            for name, preds in classical_results.items():
                all_models[name] = preds
            comp_rows = []
            for model_name, labels in all_models.items():
                vc = pd.Series(labels).value_counts()
                for sentiment in ["positive", "neutral", "negative"]:
                    comp_rows.append({
                        "model": model_name,
                        "sentiment": sentiment,
                        "count": int(vc.get(sentiment, 0)),
                    })
            comp = pd.DataFrame(comp_rows)
            fig = px.bar(comp, x="model", y="count", color="sentiment",
                         color_discrete_map=COLOR_MAP, barmode="stack",
                         title=f"Model comparison ({len(all_models)} models)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Classical models (SVM/LogReg/NB) are binary (pos/neg) — they cannot "
                       "produce a neutral label. RoBERTa is the 3-class production model.")
        else:
            missing = []
            if vectorizer is None:
                missing.append("`models/vectorizer.joblib`")
            if not classical:
                missing.append("classical model files")
            st.info(f"Model comparison unavailable: missing {', '.join(missing)}. "
                    "Run `python -m src.train_classical` from the repo root.")

    st.divider()
    st.subheader("📈 Sentiment Over Time")
    df_time = df.dropna(subset=["published"]).copy()
    if not df_time.empty:
        df_time["date"] = df_time["published"].dt.date
        ts = df_time.groupby(["date", "sentiment"]).size().reset_index(name="count")
        fig = px.area(ts, x="date", y="count", color="sentiment",
                      color_discrete_map=COLOR_MAP)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timestamps available for time-series view.")

    st.divider()
    st.subheader("💬 Top Comments")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("##### 👍 Most-liked Positive")
        for _, row in df[df["sentiment"] == "positive"].nlargest(5, "likes").iterrows():
            st.markdown(f"**{row['author']}** &nbsp;·&nbsp; 👍 {row['likes']}")
            st.caption(row["text"][:400])
            st.divider()
    with t2:
        st.markdown("##### 👎 Most-liked Negative")
        for _, row in df[df["sentiment"] == "negative"].nlargest(5, "likes").iterrows():
            st.markdown(f"**{row['author']}** &nbsp;·&nbsp; 👍 {row['likes']}")
            st.caption(row["text"][:400])
            st.divider()

    st.divider()
    csv = df.drop(columns=["score", "weight"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download results as CSV", csv,
                       file_name=f"sentiment_{video_id}.csv", mime="text/csv")

elif go and not url:
    st.warning("Paste a YouTube URL first.")
