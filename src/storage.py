# src/storage.py

import logging
from queue import Queue

logger = logging.getLogger(__name__)

def storage_worker(storage_queue: Queue):
    """
    Worker function to handle storage of embeddings.

    Args:
        storage_queue (Queue): Queue containing storage tasks.
    """
    while True:
        task = storage_queue.get()
        if task is None:
            logger.info("Storage received shutdown signal.")
            storage_queue.task_done()
            break
        song_id, embedding_file = task
        # Additional storage operations can be implemented here (e.g., uploading to a database)
        logger.info(f"Stored embedding for song ID {song_id} at {embedding_file}")
        storage_queue.task_done()
