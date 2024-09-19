# src/downloader.py

import os
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

logger = logging.getLogger(__name__)

class Downloader:
    def __init__(self, raw_dir: str, download_workers: int = 2):
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        self.download_workers = download_workers

    def download_file(self, url: str, song_id: int) -> Tuple[int, str]:
        """
        Downloads a single file from the given URL.
        """
        local_filename = os.path.join(self.raw_dir, f"{song_id}.m4a")
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            logger.info(f"Downloaded {url} to {local_filename}")
            return (song_id, local_filename)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url} for song ID {song_id}: {e}")
            return (song_id, None)

    def download_files(self, url_song_id_pairs: List[Tuple[str, int]]) -> List[Tuple[int, str]]:
        """
        Downloads multiple files concurrently.
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.download_workers) as executor:
            future_to_song = {
                executor.submit(self.download_file, url, song_id): song_id
                for url, song_id in url_song_id_pairs
            }
            for future in as_completed(future_to_song):
                song_id = future_to_song[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Unhandled exception during download of song ID {song_id}: {e}")
                    results.append((song_id, None))
        return results
