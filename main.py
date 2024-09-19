# main.py

import os
import argparse
import logging
from queue import Queue
from threading import Thread
import time
import torch
from transformers import AutoFeatureExtractor
import pandas as pd
import gc

from src.downloader import Downloader
from src.converter import Converter
from src.embedder import Embedder
from src.storage import Storage

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("pipeline.log", mode='a')
        ]
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Audio Embedding Pipeline")
    parser.add_argument('--csv', type=str, required=True, help="Path to the CSV file containing audio metadata.")
    parser.add_argument('--download_workers', type=int, default=4, help="Number of download worker threads.")
    parser.add_argument('--convert_workers', type=int, default=4, help="Number of convert worker threads.")
    parser.add_argument('--embed_workers', type=int, default=8, help="Number of embed worker threads.")
    parser.add_argument('--storage_workers', type=int, default=4, help="Number of storage worker threads.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for embedding.")
    parser.add_argument('--model_name', type=str, default="m-a-p/MERT-v1-95M", help="Name of the embedding model.")
    parser.add_argument('--max_missing_threshold', type=float, default=0.2, help="Maximum allowed fraction of missing preview URLs before proceeding.")
    return parser.parse_args()

def load_csv(csv_path: str) -> list:
    """
    Loads the CSV file and returns a list of (previewUrl, song_id) tuples.
    Ensures that only rows with valid 'previewUrl' and 'song_id' are included.
    """
    logger = logging.getLogger(__name__)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV file {csv_path}: {e}")
        return []

    # Define required columns
    required_columns = ['id', 'previewUrl']  # Adjust if 'id' is named differently
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"CSV is missing required column: {col}")
            return []

    # Rename columns if necessary
    df = df.rename(columns={'id': 'song_id'})  # Rename 'id' to 'song_id'

    # Drop rows with missing 'song_id' or 'previewUrl'
    initial_count = len(df)
    df = df.dropna(subset=['song_id', 'previewUrl'])
    final_count = len(df)
    logger.info(f"Dropped {initial_count - final_count} rows due to missing 'song_id' or 'previewUrl'.")

    # Convert 'song_id' to integer, drop rows where conversion fails
    df['song_id'] = pd.to_numeric(df['song_id'], errors='coerce')
    df = df.dropna(subset=['song_id'])
    df['song_id'] = df['song_id'].astype(int)

    # Extract the relevant columns
    url_song_id = list(zip(df['previewUrl'], df['song_id']))
    logger.info(f"Loaded {len(url_song_id)} valid song entries from CSV.")
    return url_song_id

def main():
    setup_logging()
    logger = logging.getLogger("main")
    args = parse_arguments()

    start_time = time.time()

    # Load CSV data
    url_song_id_pairs = load_csv(args.csv)
    if not url_song_id_pairs:
        logger.error("No valid song entries to process. Exiting.")
        return

    # Initialize components
    downloader = Downloader(raw_dir=os.path.join('data', 'raw'), download_workers=args.download_workers)
    converter = Converter(wav_dir=os.path.join('data', 'wav'), convert_workers=args.convert_workers)
    storage = Storage(storage_dir=os.path.join('data', 'embeddings', f"audio_embeddings_{args.model_name.replace('/', '_')}"))

    # Setup GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        logger.error("GPU is not available. Please ensure a compatible GPU is available.")
        return
    logger.info(f"Using device: {device}")
    logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Initialize Embedder
    processor = AutoFeatureExtractor.from_pretrained(args.model_name)
    embedder = Embedder(
        model_name=args.model_name,
        device=device,
        processor=processor,
        embedding_root=os.path.join('data', 'embeddings'),
        batch_size=args.batch_size
    )

    # Initialize Queues
    download_queue = Queue()
    convert_queue = Queue()
    embed_queue = Queue()
    storage_queue = Queue()

    # Populate download_queue
    for url, song_id in url_song_id_pairs:
        download_queue.put((url, song_id))

    # Define worker functions
    def download_worker_func():
        while True:
            try:
                url, song_id = download_queue.get_nowait()
            except:
                break
            result = downloader.download_file(url, song_id)
            if result[1]:  # If download was successful
                convert_queue.put((result[1], result[0]))
            download_queue.task_done()

    def convert_worker_func():
        while True:
            try:
                raw_path, song_id = convert_queue.get(timeout=5)
            except:
                break
            result = converter.convert_file(raw_path, song_id)
            if result[1]:  # If conversion was successful
                embed_queue.put((result[0], result[1]))
            convert_queue.task_done()

    # Start Download Workers
    download_threads = []
    for _ in range(args.download_workers):
        t = Thread(target=download_worker_func)
        t.start()
        download_threads.append(t)

    # Wait for downloads to finish
    download_queue.join()
    for t in download_threads:
        t.join()
    logger.info("All downloads completed.")

    # Start Convert Workers
    convert_threads = []
    for _ in range(args.convert_workers):
        t = Thread(target=convert_worker_func)
        t.start()
        convert_threads.append(t)

    # Wait for conversions to finish
    convert_queue.join()
    for t in convert_threads:
        t.join()
    logger.info("All conversions completed.")

    # Start Embed Workers
    embed_threads = []
    for _ in range(args.embed_workers):
        t = Thread(target=embedder.embed_worker, args=(embed_queue, storage_queue))
        t.start()
        embed_threads.append(t)

    # Start Storage Workers
    storage_threads = []
    for _ in range(args.storage_workers):
        t = Thread(target=storage.storage_worker, args=(storage_queue,))
        t.start()
        storage_threads.append(t)

    # Wait for embeddings to finish
    embed_queue.join()
    logger.info("All embeddings completed.")

    # Send shutdown signals to embed workers
    for _ in range(args.embed_workers):
        embed_queue.put(None)
    for t in embed_threads:
        t.join()

    # Wait for storage to finish
    storage_queue.join()
    logger.info("All storage operations completed.")

    # Send shutdown signals to storage workers
    for _ in range(args.storage_workers):
        storage_queue.put(None)
    for t in storage_threads:
        t.join()

    total_time = time.time() - start_time
    logger.info(f"Total pipeline execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
