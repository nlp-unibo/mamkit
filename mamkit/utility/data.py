"""Downloads of corpus files and debate recordings."""

import logging
from pathlib import Path
from typing import Iterable

import httpx
import yt_dlp
from tqdm import tqdm


def download(url: str, file_path: Path):
    """Downloads ``url`` to ``file_path``, with a progress bar.

    The file is written under another name and renamed when complete, so an
    interrupted download never leaves a truncated ``file_path``.
    """
    file_path = Path(file_path)
    partial = file_path.with_name(file_path.name + ".part")
    with httpx.Client() as client, client.stream("GET", url) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            partial.open("wb") as file,
            tqdm(
                desc=f"Downloading: {url}",
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress,
        ):
            for chunk in response.iter_bytes():
                file.write(chunk)
                progress.update(len(chunk))
    partial.replace(file_path)


def youtube_download(save_path: Path, debate_ids: Iterable, debate_urls: Iterable):
    """Downloads the audio of each debate video to ``<save_path>/<id>/full_audio.wav``.

    A debate whose file already exists is skipped, so a retried download
    resumes after the last complete debate.
    """
    for debate_id, url in tqdm(dict(zip(debate_ids, debate_urls)).items()):
        folder = Path(save_path).joinpath(debate_id)
        folder.mkdir(parents=True, exist_ok=True)
        filename = folder.joinpath("full_audio")
        if filename.with_suffix(".wav").exists():
            logging.info(f"Skipping {url}, since {filename.name} already exists")
            continue
        options = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": filename.as_posix(),
        }
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([url])
