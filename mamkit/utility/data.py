from pathlib import Path

import httpx
from tqdm import tqdm
import fsspec


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
                    'desc': url,
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


# TODO: integrate progress bar
def download_from_git(
        repo: str,
        org: str,
        folder: Path,
        destination: Path,
        force_download: bool = False
):
    if destination.exists() and not force_download:
        return

    fs = fsspec.filesystem('github', org=org, repo=repo)
    fs.get(str(folder.as_posix()), str(destination.as_posix()), recursive=True)
