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
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode='a')
        ]
    )

def read_csv_file(csv_path):
    try:
        # Read CSV, skip bad lines, and log warnings
        df = pd.read_csv(csv_path, on_bad_lines='warn')  # For pandas >= 1.3.0
        logging.info(f"Successfully read CSV file: {csv_path}")
        return df
    except pd.errors.ParserError as e:
        logging.error(f"ParserError while reading CSV file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error while reading CSV file: {e}")
        sys.exit(1)

def main():
    setup_logging()
    logger = logging.getLogger("Main")

    parser = argparse.ArgumentParser(description="MERT Model Audio Embedding Pipeline")
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with audio metadata')
    parser.add_argument('--download_workers', type=int, default=8, help='Number of downloader threads')
    parser.add_argument('--convert_workers', type=int, default=4, help='Number of converter threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for embedding')
    parser.add_argument('--model_name', type=str, default="m-a-p/MERT-v1-95M", help='Model name for embedding')
    parser.add_argument('--max_missing_threshold', type=float, default=0.2, help='Maximum allowed fraction of missing preview URLs before proceeding')
    args = parser.parse_args()

    # Create necessary directories
    current_dir = os.path.dirname(os.path.realpath(__file__))
    download_dir = os.path.join(current_dir, 'data', 'raw')
    wav_dir = os.path.join(current_dir, 'data', 'wav')
    embedding_root = os.path.join(current_dir, 'data', 'embeddings')
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(embedding_root, exist_ok=True)

    # Read CSV file with enhanced error handling
    csv_file = args.csv
    if not os.path.exists(csv_file):
        logger.error(f"CSV file does not exist: {csv_file}")
        sys.exit(1)

    df = read_csv_file(csv_file)
    initial_count = len(df)
    download_tasks = []
    for index, row in df.iterrows():
        song_id = row.get('id')
        preview_url = row.get('previewUrl')

        if pd.isna(preview_url):
            logger.warning(f"Missing previewUrl for song ID {song_id} at row {index + 2}")  # +2 accounts for header and 0-indexing
            continue

        # Basic validation of preview_url
        if not isinstance(preview_url, str) or not preview_url.startswith(('http://', 'https://')):
            logger.warning(f"Invalid previewUrl for song ID {song_id} at row {index + 2}: {preview_url}")
            continue

        download_tasks.append((song_id, preview_url))

    missing_preview_count = initial_count - len(download_tasks)
    logger.info(f"Total download tasks: {len(download_tasks)}")
    logger.info(f"Total missing or invalid previewUrls: {missing_preview_count}")

    # Optional: Alert if missing previews exceed a threshold
    if missing_preview_count / initial_count > args.max_missing_threshold:
        logger.error(f"High number of missing previewUrls: {missing_preview_count} out of {initial_count} ({missing_preview_count / initial_count * 100:.2f}%)")
        # Instead of exiting, decide to proceed or handle accordingly
        # For now, proceed with a warning

    if not download_tasks:
        logger.error("No valid download tasks found.")
        sys.exit(1)

    # Initialize queues
    download_queue = Queue(maxsize=100)
    convert_queue = Queue(maxsize=100)
    embed_queue = Queue(maxsize=100)
    storage_queue = Queue(maxsize=100)

    # Start downloader threads
    num_downloader_threads = args.download_workers
    downloader_threads = []
    for _ in range(num_downloader_threads):
        t = Thread(target=downloader_worker, args=(download_queue, convert_queue, download_dir, args.download_workers))
        t.start()
        downloader_threads.append(t)

    # Start converter threads
    num_converter_threads = args.convert_workers
    converter_threads = []
    for _ in range(num_converter_threads):
        t = Thread(target=converter_worker, args=(convert_queue, embed_queue, wav_dir, args.convert_workers))
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

    # Add sentinel values to downloader queue to signal no more tasks
    for _ in range(num_downloader_threads):
        download_queue.put(None)

    # Wait for all downloads to finish
    for t in downloader_threads:
        t.join()
    logger.info("All downloads completed.")

    # Add sentinel values to converter queue to signal no more tasks
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

    # Add sentinel to storage queue to signal no more data
    storage_queue.put(None)

    # Wait for storage to finish
    storage_thread.join()
    logger.info("All embeddings stored.")

    logger.info("Processing complete.")

if __name__ == "__main__":
    main()
