# src/converter.py

import os
import subprocess
import logging
from queue import Queue
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

def convert_m4a_to_wav(song_id, m4a_path, wav_dir):
    """
    Converts an M4A file to WAV format using ffmpeg.

    Args:
        song_id (int): The unique identifier for the song.
        m4a_path (str): Path to the source M4A file.
        wav_dir (str): Directory to save the converted WAV file.

    Returns:
        tuple: (song_id, wav_path) if successful, else None.
    """
    try:
        wav_path = os.path.join(wav_dir, f"{song_id}.wav")
        command = ['ffmpeg', '-y', '-i', m4a_path, wav_path]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Converted {m4a_path} to {wav_path}")
        return (song_id, wav_path)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert {m4a_path} to wav: {e.stderr.decode()}")
        return None

def converter_worker(convert_queue: Queue, embed_queue: Queue, wav_dir: str, max_workers: int = 4):
    """
    Worker function to convert M4A files to WAV using a process pool.

    Args:
        convert_queue (Queue): Queue containing conversion tasks.
        embed_queue (Queue): Queue to enqueue successfully converted files for embedding.
        wav_dir (str): Directory to save converted WAV files.
        max_workers (int): Number of processes in the process pool.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        while True:
            task = convert_queue.get()
            if task is None:
                logger.info("Converter received shutdown signal.")
                convert_queue.task_done()
                break
            song_id, m4a_path = task
            future = executor.submit(convert_m4a_to_wav, song_id, m4a_path, wav_dir)
            result = future.result()
            if result:
                embed_queue.put(result)
            convert_queue.task_done()
