# main.py

import sys
import os
import logging
from queue import Queue
from threading import Thread
import argparse
import pandas as pd

from src.downloader import downloader_worker
from src.converter import converter_worker
from src.embedder import Embedder
from src.storage import storage_worker

import torch
from transformers import AutoFeatureExtractor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger("Main")

    parser = argparse.ArgumentParser(description="MERT Model Audio Embedding Pipeline")
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with audio metadata')
    parser.add_argument('--download_workers', type=int, default=8, help='Number of downloader threads')
    parser.add_argument('--convert_workers', type=int, default=4, help='Number of converter threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for embedding')
    parser.add_argument('--model_name', type=str, default="m-a-p/MERT-v1-95M", help='Model name for embedding')
    args = parser.parse_args()

    # Create necessary directories
    current_dir = os.path.dirname(os.path.realpath(__file__))
    download_dir = os.path.join(current_dir, 'data', 'raw')
    wav_dir = os.path.join(current_dir, 'data', 'wav')
    embedding_root = os.path.join(current_dir, 'data', 'embeddings')
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(embedding_root, exist_ok=True)

    # Read CSV file
    csv_file = args.csv
    if not os.path.exists(csv_file):
        logger.error(f"CSV file does not exist: {csv_file}")
        sys.exit(1)

    # Read the CSV to get download tasks
    df = pd.read_csv(csv_file)
    download_tasks = []
    for _, row in df.iterrows():
        song_id = row['id']
        preview_url = row.get('previewUrl', None)
        if pd.isna(preview_url):
            logger.warning(f"Missing previewUrl for song ID {song_id}")
            continue
        download_tasks.append((song_id, preview_url))

    if not download_tasks:
        logger.error("No download tasks found.")
        sys.exit(1)

    logger.info(f"Total download tasks: {len(download_tasks)}")

    # Initialize queues
    download_queue = Queue(maxsize=100)
    convert_queue = Queue(maxsize=100)
    embed_queue = Queue(maxsize=100)
    storage_queue = Queue(maxsize=100)

    # Start downloader threads
    num_downloader_threads = args.download_workers
    downloader_threads = []
    for _ in range(num_downloader_threads):
        t = Thread(target=downloader_worker, args=(download_queue, convert_queue, download_dir))
        t.start()
        downloader_threads.append(t)

    # Start converter threads
    num_converter_threads = args.convert_workers
    converter_threads = []
    for _ in range(num_converter_threads):
        t = Thread(target=converter_worker, args=(convert_queue, embed_queue, wav_dir))
        t.start()
        converter_threads.append(t)

    # Initialize Embedder
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    model_name = args.model_name
    processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    embedder = Embedder(model_name, device, processor, embedding_root, batch_size=args.batch_size)

    # Start storage thread
    storage_thread = Thread(target=storage_worker, args=(storage_queue,))
    storage_thread.start()

    # Start embedder thread
    embedder_thread = Thread(target=embedder.embed_worker, args=(embed_queue, storage_queue))
    embedder_thread.start()

    # Enqueue download tasks
    for task in download_tasks:
        download_queue.put(task)

    # Add sentinel values to downloader queue
    for _ in range(num_downloader_threads):
        download_queue.put(None)

    # Wait for all downloads to finish
    for t in downloader_threads:
        t.join()
    logger.info("All downloads completed.")

    # Add sentinel values to converter queue
    for _ in range(num_converter_threads):
        convert_queue.put(None)

    # Wait for all conversions to finish
    for t in converter_threads:
        t.join()
    logger.info("All conversions completed.")

    # Add sentinel to embed queue to signal no more data
    embed_queue.put(None)

    # Wait for embedder to finish
    embedder_thread.join()
    logger.info("All embeddings completed.")

    # Add sentinel to storage queue
    storage_queue.put(None)

    # Wait for storage to finish
    storage_thread.join()
    logger.info("All embeddings stored.")

    logger.info("Processing complete.")

if __name__ == "__main__":
    main()
