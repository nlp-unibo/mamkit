"""Downloads write complete files or nothing."""

import httpx
import pytest

from mamkit.utility import data


def serve(monkeypatch, chunks):
    """Makes every ``httpx.Client`` answer with ``chunks``, raising on errors."""

    def stream():
        for chunk in chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def handler(request):
        return httpx.Response(200, content=stream())

    client = httpx.Client
    monkeypatch.setattr(
        data.httpx,
        "Client",
        lambda: client(transport=httpx.MockTransport(handler)),
    )


def test_download_writes_the_whole_file(monkeypatch, tmp_path):
    serve(monkeypatch, [b"ab", b"cd"])
    data.download("https://example.org/file", tmp_path / "file")
    assert (tmp_path / "file").read_bytes() == b"abcd"
    assert [p.name for p in tmp_path.iterdir()] == ["file"]


def test_interrupted_download_leaves_no_file(monkeypatch, tmp_path):
    serve(monkeypatch, [b"ab", OSError("connection reset")])
    with pytest.raises(OSError, match="connection reset"):
        data.download("https://example.org/file", tmp_path / "file")
    assert not (tmp_path / "file").exists()


def test_youtube_download_skips_debates_already_downloaded(monkeypatch, tmp_path):
    downloaded = []

    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            downloaded.extend(urls)

    monkeypatch.setattr(data.yt_dlp, "YoutubeDL", Downloader)
    (tmp_path / "d1").mkdir()
    (tmp_path / "d1" / "full_audio.wav").write_bytes(b"")
    data.youtube_download(tmp_path, ["d1", "d2"], ["u1", "u2"])
    assert downloaded == ["u2"]
