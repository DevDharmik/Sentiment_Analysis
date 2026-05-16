"""
YouTube Comment Sentiment Analysis Dashboard
Self-contained app — uses YouTube Data API v3 + HuggingFace RoBERTa.
Optionally compares against the SVM model from models/svm_tfidf.joblib.
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
SVM_PATH = "models/svm_tfidf.joblib"

ROBERTA_LABEL_MAP = {
    "label_0": "negative", "label_1": "neutral", "label_2": "positive",
    "negative": "negative", "neutral": "neutral", "positive": "positive",
}
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
def get_svm_model():
    if not HAS_JOBLIB or not os.path.exists(SVM_PATH):
        return None
    try:
        return joblib.load(SVM_PATH)
    except Exception:
        return None


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


def score_svm(texts, model):
    if not model or not texts:
        return []
    preds = model.predict(texts)
    return [str(p).lower() for p in preds]


# ---------- UI ----------
st.title("🎬 YouTube Comment Sentiment Analysis")
st.caption("Paste any YouTube URL → get sentiment breakdown in seconds.")

if not API_KEY:
    st.error("`YOUTUBE_API_KEY` env var is not set. Add it as a Codespaces secret and rebuild the container.")
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

    # Video header
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

    # Fetch
    with st.spinner(f"Fetching up to {max_comments} comments..."):
        comments = fetch_comments(youtube, video_id, max_comments)
    if not comments:
        st.warning("No comments fetched (comments might be disabled).")
        st.stop()
    st.success(f"Fetched {len(comments)} comments.")

    # Score with RoBERTa
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

    # Optional SVM scoring
    svm_model = get_svm_model()
    if svm_model is not None:
        try:
            df["svm_sentiment"] = score_svm(texts, svm_model)
        except Exception as e:
            st.info(f"SVM model loaded but scoring failed: {e}")

    # ---------- Metrics ----------
    st.divider()
    st.subheader("📊 Sentiment Breakdown")

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

    # ---------- Charts ----------
    g1, g2 = st.columns(2)
    with g1:
        fig = px.pie(names=counts.index, values=counts.values, hole=0.5,
                     color=counts.index, color_discrete_map=COLOR_MAP,
                     title="Sentiment distribution (RoBERTa)")
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        if "svm_sentiment" in df.columns:
            comp = pd.DataFrame({
                "RoBERTa": df["sentiment"].value_counts(),
                "SVM":     df["svm_sentiment"].value_counts(),
            }).fillna(0).reset_index().rename(columns={"index": "sentiment"})
            comp = comp.melt(id_vars="sentiment", var_name="model", value_name="count")
            fig = px.bar(comp, x="sentiment", y="count", color="model",
                         barmode="group", title="Model comparison: RoBERTa vs SVM")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SVM model not found at `models/svm_tfidf.joblib`. "
                    "Run `python -m src.train_classical` to enable the comparison view.")

    # Sentiment over time
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

    # Top comments
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

    # Download
    st.divider()
    csv = df.drop(columns=["score", "weight"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download results as CSV", csv,
                       file_name=f"sentiment_{video_id}.csv", mime="text/csv")

elif go and not url:
    st.warning("Paste a YouTube URL first.")
