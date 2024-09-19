# src/storage.py

import os
import logging
from queue import Queue
from typing import Tuple

logger = logging.getLogger(__name__)

class Storage:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def store_embedding(self, song_id: int, embedding_file: str):
        """
        Stores the embedding file. This function can be extended to handle more complex storage mechanisms.
        """
        try:
            # For simplicity, we're just ensuring the file exists in the storage directory
            # You can add more logic here (e.g., moving files, uploading to cloud storage)
            os.makedirs(self.storage_dir, exist_ok=True)
            # If you need to move or copy the file, uncomment the following lines:
            # import shutil
            # shutil.move(embedding_file, self.storage_dir)
            logger.info(f"Stored embedding for song ID {song_id} at {embedding_file}")
        except Exception as e:
            logger.error(f"Failed to store embedding for song ID {song_id}: {e}")

    def storage_worker(self, storage_queue: Queue):
        while True:
            task = storage_queue.get()
            if task is None:
                logger.info("Storage received shutdown signal.")
                storage_queue.task_done()
                break
            song_id, embedding_file = task
            self.store_embedding(song_id, embedding_file)
            storage_queue.task_done()
