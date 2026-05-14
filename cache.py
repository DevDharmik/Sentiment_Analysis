"""
src/cache.py — SQLite cache for analysed YouTube videos.

When a video is analysed once, all of its metadata, comments, and
model predictions are persisted here. Re-visiting the same video
becomes a single SQL query instead of a YouTube API call + model
inference for every comment.

Schema:
- videos: one row per video (metadata + channel info + fetched_at timestamp)
- comments: one row per comment (text + all 4 model predictions)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_DB_PATH = "sentiment_analysis.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS videos (
    video_id              TEXT PRIMARY KEY,
    title                 TEXT,
    channel_id            TEXT,
    channel_title         TEXT,
    channel_thumbnail     TEXT,
    channel_subscribers   INTEGER,
    channel_video_count   INTEGER,
    published_at          TEXT,
    view_count            INTEGER,
    like_count            INTEGER,
    comment_count         INTEGER,
    duration_iso          TEXT,
    thumbnail_url         TEXT,
    fetched_at            TEXT
);

CREATE TABLE IF NOT EXISTS comments (
    comment_id            TEXT PRIMARY KEY,
    video_id              TEXT NOT NULL,
    author                TEXT,
    text                  TEXT,
    clean_text            TEXT,
    language              TEXT,
    like_count            INTEGER,
    reply_count           INTEGER,
    published_at          TEXT,
    svm_pred              TEXT,
    logreg_pred           TEXT,
    nb_pred               TEXT,
    transformer_pred      TEXT,
    transformer_raw       TEXT,
    transformer_confidence REAL,
    transformer_score     REAL,
    is_uncertain          INTEGER,
    FOREIGN KEY (video_id) REFERENCES videos (video_id)
);

CREATE INDEX IF NOT EXISTS idx_comments_video ON comments(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_likes ON comments(like_count DESC);
"""


# ---------------------------------------------------------------------------
# Connection & setup
# ---------------------------------------------------------------------------
def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection, ensuring schema exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True) if "/" in db_path else None
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA_SQL)
    con.commit()
    return con


# ---------------------------------------------------------------------------
# Save: video + channel
# ---------------------------------------------------------------------------
def save_video(con: sqlite3.Connection, video_meta, channel_info) -> None:
    """Upsert a video record. Accepts VideoMeta + ChannelInfo dataclasses."""
    row = {
        "video_id": video_meta.video_id,
        "title": video_meta.title,
        "channel_id": video_meta.channel_id,
        "channel_title": video_meta.channel_title,
        "channel_thumbnail": channel_info.thumbnail_url,
        "channel_subscribers": channel_info.subscriber_count,
        "channel_video_count": channel_info.video_count,
        "published_at": video_meta.published_at,
        "view_count": video_meta.view_count,
        "like_count": video_meta.like_count,
        "comment_count": video_meta.comment_count,
        "duration_iso": video_meta.duration_iso,
        "thumbnail_url": video_meta.thumbnail_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    cols = ",".join(row.keys())
    placeholders = ",".join("?" * len(row))
    con.execute(
        f"INSERT OR REPLACE INTO videos ({cols}) VALUES ({placeholders})",
        tuple(row.values()),
    )
    con.commit()


# ---------------------------------------------------------------------------
# Save: comments with predictions
# ---------------------------------------------------------------------------
def save_comments(con: sqlite3.Connection, video_id: str, df: pd.DataFrame) -> int:
    """
    Save analysed comments to the cache.
    df must have these columns (after analysis):
      comment_id, author, text, clean_text, language,
      like_count, reply_count, published_at,
      svm_pred, logreg_pred, nb_pred,
      transformer_pred, transformer_raw, transformer_confidence,
      transformer_score, is_uncertain
    Returns number of rows written.
    """
    if df.empty:
        return 0
    df = df.copy()
    df["video_id"] = video_id
    # SQLite stores booleans as integers
    if "is_uncertain" in df.columns:
        df["is_uncertain"] = df["is_uncertain"].astype(int)

    cols = [
        "comment_id", "video_id", "author", "text", "clean_text", "language",
        "like_count", "reply_count", "published_at",
        "svm_pred", "logreg_pred", "nb_pred",
        "transformer_pred", "transformer_raw",
        "transformer_confidence", "transformer_score", "is_uncertain",
    ]
    rows = df[cols].to_dict(orient="records")
    placeholders = ",".join("?" * len(cols))
    col_list = ",".join(cols)

    con.executemany(
        f"INSERT OR REPLACE INTO {('comments')} ({col_list}) VALUES ({placeholders})",
        [tuple(r[c] for c in cols) for r in rows],
    )
    con.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Load: check cache freshness and retrieve
# ---------------------------------------------------------------------------
def is_cached(con: sqlite3.Connection, video_id: str) -> bool:
    """Return True if this video has any cached data."""
    cur = con.execute("SELECT 1 FROM videos WHERE video_id = ? LIMIT 1", (video_id,))
    return cur.fetchone() is not None


def cache_age_hours(con: sqlite3.Connection, video_id: str) -> Optional[float]:
    """Hours since this video was last cached. None if never cached."""
    cur = con.execute(
        "SELECT fetched_at FROM videos WHERE video_id = ?", (video_id,)
    )
    row = cur.fetchone()
    if not row:
        return None
    fetched_at = datetime.fromisoformat(row[0])
    age = datetime.now(timezone.utc) - fetched_at
    return age.total_seconds() / 3600.0


def is_fresh(con: sqlite3.Connection, video_id: str, max_age_hours: float = 24) -> bool:
    """True if video is cached AND younger than max_age_hours."""
    age = cache_age_hours(con, video_id)
    return age is not None and age < max_age_hours


def load_video(con: sqlite3.Connection, video_id: str) -> Optional[dict]:
    """Get a cached video's metadata + channel info as a dict."""
    cur = con.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def load_comments(con: sqlite3.Connection, video_id: str) -> pd.DataFrame:
    """Get all cached comments for a video as a DataFrame."""
    df = pd.read_sql_query(
        "SELECT * FROM comments WHERE video_id = ?",
        con,
        params=(video_id,),
    )
    if not df.empty and "is_uncertain" in df.columns:
        df["is_uncertain"] = df["is_uncertain"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# Stats & cleanup
# ---------------------------------------------------------------------------
def cache_stats(con: sqlite3.Connection) -> dict:
    """Quick summary: how many videos + comments are cached."""
    n_videos = con.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    n_comments = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    return {"videos_cached": n_videos, "comments_cached": n_comments}


def clear_video(con: sqlite3.Connection, video_id: str) -> None:
    """Delete a single video and its comments from the cache."""
    con.execute("DELETE FROM comments WHERE video_id = ?", (video_id,))
    con.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
    con.commit()


def clear_all(con: sqlite3.Connection) -> None:
    """Nuke the cache. Useful for testing."""
    con.execute("DELETE FROM comments")
    con.execute("DELETE FROM videos")
    con.commit()
