from src.extractor import parse_video_id


def test_parse_video_id_accepts_supported_youtube_urls():
    video_id = "dQw4w9WgXcQ"

    assert parse_video_id(video_id) == video_id
    assert parse_video_id(f"https://youtu.be/{video_id}?si=test") == video_id
    assert parse_video_id(f"https://www.youtube.com/watch?v={video_id}") == video_id
    assert parse_video_id(f"youtube.com/shorts/{video_id}") == video_id
    assert parse_video_id(f"https://music.youtube.com/watch?v={video_id}") == video_id


def test_parse_video_id_rejects_lookalike_hosts_and_invalid_ids():
    video_id = "dQw4w9WgXcQ"

    assert (
        parse_video_id(f"https://youtube.com.example.test/watch?v={video_id}") is None
    )
    assert parse_video_id(f"https://example.test/{video_id}") is None
    assert parse_video_id("https://youtu.be/not-an-id") is None
