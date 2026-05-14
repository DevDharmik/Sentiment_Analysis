"""
src/extractor.py — YouTube Data API v3 wrapper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import retry, stop_after_attempt, wait_exponential


_VIDEO_ID_PATTERNS = [
    r"(?:v=|/)([0-9A-Za-z_-]{11}).*",
    r"^([0-9A-Za-z_-]{11})$",
]


def parse_video_id(url_or_id: str) -> Optional[str]:
    url_or_id = url_or_id.strip()
    for pattern in _VIDEO_ID_PATTERNS:
        m = re.search(pattern, url_or_id)
        if m:
            return m.group(1)
    return None


@dataclass
class VideoMeta:
    video_id: str
    title: str
    channel_id: str
    channel_title: str
    published_at: str
    view_count: int
    like_count: int
    comment_count: int
    thumbnail_url: str
    duration_iso: str


@dataclass
class ChannelInfo:
    channel_id: str
    title: str
    description: str
    subscriber_count: int
    video_count: int
    view_count: int
    thumbnail_url: str
    country: str


@dataclass
class Comment:
    comment_id: str
    author: str
    text: str
    like_count: int
    reply_count: int
    published_at: str


class CommentsDisabledError(Exception):
    pass


class VideoNotFoundError(Exception):
    pass


class QuotaExceededError(Exception):
    pass


def get_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def fetch_video_meta(client, video_id: str) -> VideoMeta:
    resp = (
        client.videos()
        .list(part="snippet,statistics,contentDetails", id=video_id)
        .execute()
    )
    if not resp.get("items"):
        raise VideoNotFoundError(f"Video {video_id} not found")
    item = resp["items"][0]
    s, st, cd = item["snippet"], item["statistics"], item["contentDetails"]
    thumb = s.get("thumbnails", {}).get("high", {}).get("url", "")
    return VideoMeta(
        video_id=video_id,
        title=s["title"],
        channel_id=s["channelId"],
        channel_title=s["channelTitle"],
        published_at=s["publishedAt"],
        view_count=int(st.get("viewCount", 0)),
        like_count=int(st.get("likeCount", 0)),
        comment_count=int(st.get("commentCount", 0)),
        thumbnail_url=thumb,
        duration_iso=cd.get("duration", ""),
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def fetch_channel_info(client, channel_id: str) -> ChannelInfo:
    resp = (
        client.channels()
        .list(part="snippet,statistics", id=channel_id)
        .execute()
    )
    if not resp.get("items"):
        raise VideoNotFoundError(f"Channel {channel_id} not found")
    item = resp["items"][0]
    s, st = item["snippet"], item["statistics"]
    sub_count_raw = st.get("subscriberCount")
    sub_count = int(sub_count_raw) if sub_count_raw is not None else -1
    return ChannelInfo(
        channel_id=channel_id,
        title=s["title"],
        description=s.get("description", "")[:300],
        subscriber_count=sub_count,
        video_count=int(st.get("videoCount", 0)),
        view_count=int(st.get("viewCount", 0)),
        thumbnail_url=s.get("thumbnails", {}).get("default", {}).get("url", ""),
        country=s.get("country", ""),
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _comments_page(client, video_id, page_token, page_size):
    return (
        client.commentThreads()
        .list(
            part="snippet",
            videoId=video_id,
            maxResults=page_size,
            pageToken=page_token,
            textFormat="plainText",
            order="relevance",
        )
        .execute()
    )


def fetch_comments(client, video_id: str, max_total: int = 500) -> list:
    rows = []
    page_token = None
    while len(rows) < max_total:
        page_size = min(100, max_total - len(rows))
        try:
            resp = _comments_page(client, video_id, page_token, page_size)
        except HttpError as e:
            msg = str(e).lower()
            if "commentsdisabled" in msg or "disabled comments" in msg:
                raise CommentsDisabledError(f"Comments disabled on {video_id}") from e
            if "quotaexceeded" in msg or "quota" in msg:
                raise QuotaExceededError("Daily YouTube API quota exceeded") from e
            raise
        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            rows.append(Comment(
                comment_id=item["snippet"]["topLevelComment"]["id"],
                author=top.get("authorDisplayName", "Anonymous"),
                text=top.get("textDisplay", ""),
                like_count=int(top.get("likeCount", 0)),
                reply_count=int(item["snippet"].get("totalReplyCount", 0)),
                published_at=top.get("publishedAt", ""),
            ))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return rows


def engagement_rate(view_count: int, like_count: int, comment_count: int) -> float:
    if view_count <= 0:
        return 0.0
    return (like_count + comment_count) / view_count


def parse_iso_duration(iso: str) -> str:
    if not iso or not iso.startswith("PT"):
        return iso
    h = re.search(r"(\d+)H", iso)
    m = re.search(r"(\d+)M", iso)
    s = re.search(r"(\d+)S", iso)
    hours = int(h.group(1)) if h else 0
    mins = int(m.group(1)) if m else 0
    secs = int(s.group(1)) if s else 0
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"
