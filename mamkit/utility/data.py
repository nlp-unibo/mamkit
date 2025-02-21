import logging
import os
from pathlib import Path
from typing import List

import httpx
import yt_dlp
from tqdm import tqdm


# kudos to: https://pub.aimind.so/download-large-file-in-python-with-beautiful-progress-bar-f4f86b394ad7
def download(
        url: str,
        file_path: Path
):
    with file_path.open('wb') as f:
        with httpx.Client() as client:
            with client.stream('GET', url) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))

                tqdm_params = {
                    'desc': f'Downloading: {url}',
                    'total': total,
                    'miniters': 1,
                    'unit': 'B',
                    'unit_scale': True,
                    'unit_divisor': 1024,
                }

                with tqdm(**tqdm_params) as pb:
                    downloaded = r.num_bytes_downloaded
                    for chunk in r.iter_bytes():
                        pb.update(r.num_bytes_downloaded - downloaded)
                        f.write(chunk)
                        downloaded = r.num_bytes_downloaded


def youtube_download(
        save_path: Path,
        debate_ids: List,
        debate_urls: List
) -> None:
    """

    :param save_path: path where to download youtube videos
    :param debate_ids: list of strings representing debates IDs
    :param debate_urls: list of strings representing the urls to the YouTube videos of the debates
    :return: None. The function populates the folder 'files/debates_audio_recordings' by creating a folder for each
             debate. Each folder contains the audio file extracted from the corresponding video
    """

    map_debate_link = dict(zip(debate_ids, debate_urls))
    for doc, link in tqdm(map_debate_link.items()):
        doc_path = save_path.joinpath(doc)
        if not save_path.exists():
            save_path.mkdir(parents=True)

        doc_path.mkdir(parents=True, exist_ok=True)
        filename = doc_path.joinpath("full_audio")

        if filename.with_suffix('.wav').exists():
            logging.info(f'Skipping {link} since {filename.name} already exists...')
            continue
        else:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': filename.as_posix()
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([link])
            except yt_dlp.utils.DownloadError as e:
                logging.info(f'Could not download {link}')
                raise e
            finally:
                os.system("youtube-dl --rm-cache-dir")
