import sys
import os
import logging
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoFeatureExtractor, AutoModel
import torchaudio
import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import io
import csv
from tqdm import tqdm
from queue import Queue
from threading import Thread
from pydub import AudioSegment
import tempfile

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("Main")

def download_and_convert(task, wav_dir, target_sr=24000):
    song_id, preview_url = task
    wav_path = os.path.join(wav_dir, f"{song_id}.wav")

    if not os.path.exists(wav_path):
        try:
            response = requests.get(preview_url)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as temp_m4a:
                temp_m4a.write(response.content)
                temp_m4a_path = temp_m4a.name

            audio = AudioSegment.from_file(temp_m4a_path, format="m4a")
            audio = audio.set_frame_rate(target_sr)
            audio.export(wav_path, format="wav")

            os.unlink(temp_m4a_path)

        except Exception as e:
            logging.error(f"Error processing {song_id}: {str(e)}")
            return None
    
    return song_id, wav_path

def read_csv_robust(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = []
        for row in reader:
            if len(row) >= 2:
                data.append(row[:len(headers)])
    return pd.DataFrame(data, columns=headers)

class PipelineDataset(IterableDataset):
    def __init__(self, csv_dir, wav_dir, processor, num_workers):
        self.csv_dir = csv_dir
        self.wav_dir = wav_dir
        self.processor = processor
        self.num_workers = num_workers
        self.download_queue = Queue(maxsize=1000)
        self.process_queue = Queue(maxsize=1000)

    def csv_reader(self):
        csv_files = [f for f in os.listdir(self.csv_dir) if f.startswith("chunk_") and f.endswith(".csv")]
        for chunk_file in csv_files:
            chunk_path = os.path.join(self.csv_dir, chunk_file)
            try:
                chunk_df = read_csv_robust(chunk_path)
                for _, row in chunk_df.iterrows():
                    if pd.notna(row['previewUrl']):
                        self.download_queue.put((row['id'], row['previewUrl']))
            except Exception as e:
                logging.error(f"Error reading {chunk_file}: {str(e)}")
        
        for _ in range(self.num_workers):
            self.download_queue.put(None)

    def downloader(self):
        while True:
            item = self.download_queue.get()
            if item is None:
                break
            result = download_and_convert(item, self.wav_dir)
            if result:
                self.process_queue.put(result)
        self.process_queue.put(None)

    def processor_worker(self):
        target_sr = self.processor.sampling_rate
        resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=target_sr)
        
        while True:
            item = self.process_queue.get()
            if item is None:
                break
            song_id, wav_path = item
            waveform, sample_rate = torchaudio.load(wav_path)
            
            if sample_rate != target_sr:
                waveform = resampler(waveform)
            
            # Ensure mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Normalize the waveform
            waveform = waveform - waveform.mean()
            waveform = waveform / waveform.abs().max()
            
            inputs = self.processor(waveform.numpy(), sampling_rate=target_sr, return_tensors="pt")
            yield song_id, inputs.input_values.squeeze(0)

    def __iter__(self):
        csv_thread = Thread(target=self.csv_reader)
        csv_thread.start()

        download_threads = []
        for _ in range(self.num_workers):
            t = Thread(target=self.downloader)
            t.start()
            download_threads.append(t)

        yield from self.processor_worker()

        csv_thread.join()
        for t in download_threads:
            t.join()

def collate_fn(batch):
    ids, audio = zip(*batch)
    return list(ids), torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)

def main():
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="Pipelined MERT Model Audio Embedding Pipeline")
    parser.add_argument('--csv_dir', type=str, required=True, help='Path to directory containing CSV files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding')
    parser.add_argument('--model_name', type=str, default="m-a-p/MERT-v1-95M", help='Model name for embedding')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    wav_dir = os.path.join(data_dir, 'raw')
    csv_dir = os.path.join(data_dir, 'csvs')
    embeddings_dir = os.path.join(data_dir, 'embeddings')
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    processor = AutoFeatureExtractor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device)

    dataset = PipelineDataset(csv_dir, wav_dir, processor, args.num_workers)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    embeddings = {}
    model.eval()
    with torch.no_grad():
        for batch_ids, batch_audio in tqdm(dataloader, desc="Generating embeddings"):
            batch_audio = batch_audio.to(device)
            outputs = model(batch_audio)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            for id, embedding in zip(batch_ids, batch_embeddings):
                embeddings[id] = embedding.cpu().numpy()

    logger.info("Embedding generation complete. Updating CSV files...")

    # Update CSV files with embeddings
    csv_files = [f for f in os.listdir(csv_dir) if f.startswith("chunk_") and f.endswith(".csv")]
    for chunk_file in tqdm(csv_files, desc="Updating CSV files with embeddings"):
        chunk_path = os.path.join(csv_dir, chunk_file)
        try:
            chunk_df = read_csv_robust(chunk_path)
            
            chunk_df_with_embeddings = chunk_df.copy()
            chunk_df_with_embeddings['embedding'] = chunk_df_with_embeddings['id'].map(embeddings)
            
            new_file_name = chunk_file.replace(".csv", "_with_embeddings.csv")
            new_file_path = os.path.join(embeddings_dir, new_file_name)
            chunk_df_with_embeddings.to_csv(new_file_path, index=False)
        except Exception as e:
            logger.error(f"Error processing {chunk_file}: {str(e)}")

    logger.info("Processing complete. Created new CSV files with embeddings in the 'data/embeddings' directory.")

if __name__ == "__main__":
    main()
