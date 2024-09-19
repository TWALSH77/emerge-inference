# src/downloader.py

import os
import requests
import logging
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

def download_file(session, song_id, url, save_dir):
    """
    Downloads a file from the given URL and saves it to the specified directory.
    
    Args:
        session (requests.Session): The requests session object.
        song_id (int): The unique identifier for the song.
        url (str): The URL to download the file from.
        save_dir (str): The directory to save the downloaded file.
    
    Returns:
        tuple: (song_id, save_path) if successful, else None.
    """
    try:
        if not url:
            logger.error(f"No URL provided for song ID {song_id}")
            return None
        save_path = os.path.join(save_dir, f"{song_id}.m4a")
        with session.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
        logger.info(f"Downloaded {url} to {save_path}")
        return (song_id, save_path)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return None

def downloader_worker(download_queue: Queue, convert_queue: Queue, download_dir: str, max_workers: int = 8):
    """
    Worker function to download files using a thread pool.

    Args:
        download_queue (Queue): Queue containing download tasks.
        convert_queue (Queue): Queue to enqueue successfully downloaded files for conversion.
        download_dir (str): Directory to save downloaded files.
        max_workers (int): Number of threads in the thread pool.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor, requests.Session() as session:
        while True:
            task = download_queue.get()
            if task is None:
                logger.info("Downloader received shutdown signal.")
                download_queue.task_done()
                break
            song_id, url = task
            future = executor.submit(download_file, session, song_id, url, download_dir)
            result = future.result()
            if result:
                convert_queue.put(result)
            download_queue.task_done()
