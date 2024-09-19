# src/embedder.py

import os
import h5py
import torch
import torchaudio
import logging
from transformers import AutoFeatureExtractor, AutoModel
from queue import Queue
from typing import List, Tuple
import time
import gc

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str, device: torch.device, processor: AutoFeatureExtractor, embedding_root: str, batch_size: int = 1):
        self.model_name = model_name
        self.device = device
        self.processor = processor
        self.embedding_root = embedding_root
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    def embed_batch(self, batch: List[Tuple[int, str]]):
        batch_start_time = time.time()
        try:
            song_ids, wav_paths = zip(*batch)
            waveforms = []
            valid_song_ids = []
            valid_wav_paths = []
            for song_id, wav_path in zip(song_ids, wav_paths):
                try:
                    waveform, sample_rate = torchaudio.load(wav_path)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    waveform = waveform.squeeze().numpy()
                    if waveform.ndim == 0:
                        waveform = waveform.reshape(1)
                    waveforms.append(waveform)
                    valid_song_ids.append(song_id)
                    valid_wav_paths.append(wav_path)
                except Exception as e:
                    logger.warning(f"Failed to load WAV file {wav_path} for song ID {song_id}: {e}")

            if not waveforms:
                logger.warning("No valid WAV files to embed in this batch.")
                return []

            inputs = self.processor(
                waveforms,
                sampling_rate=self.processor.sampling_rate,
                padding=True,
                return_tensors="pt"
            )

            # Remove the third dimension if exists (for compatibility)
            for k, v in inputs.items():
                if v.dim() == 4:
                    inputs[k] = v.squeeze(2)

            # Move inputs to GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            with torch.inference_mode():
                outputs = self.model(**inputs, output_hidden_states=True)

            all_layer_hidden_states = torch.stack(outputs.hidden_states).permute(1, 0, 2, 3)
            time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
            embeddings = time_reduced_hidden_states.mean(dim=1).cpu().numpy()

            embedding_files = []
            for song_id, embedding, wav_path in zip(valid_song_ids, embeddings, valid_wav_paths):
                embedding_dir = os.path.join(self.embedding_root, f"audio_embeddings_{self.model_name.replace('/', '_')}")
                os.makedirs(embedding_dir, exist_ok=True)
                embedding_file = os.path.join(embedding_dir, f"audio_embeddings_{self.model_name.replace('/', '_')}_{song_id}.h5")
                try:
                    with h5py.File(embedding_file, 'w') as hf:
                        hf.create_dataset(str(song_id), data=embedding)
                    embedding_files.append((song_id, embedding_file))
                    logger.info(f"Embedded and saved {wav_path} to {embedding_file}")
                except Exception as e:
                    logger.error(f"Failed to save embedding for song ID {song_id}: {e}")

            batch_time = time.time() - batch_start_time
            logger.info(f"Batch embedding time: {batch_time:.2f} seconds for {len(batch)} items.")

            # Clear memory
            del inputs, outputs, embeddings
            gc.collect()
            torch.cuda.empty_cache()

            return embedding_files
        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            return []

    def embed_worker(self, embed_queue: Queue, storage_queue: Queue):
        batch = []
        while True:
            task = embed_queue.get()
            if task is None:
                logger.info("Embedder received shutdown signal.")
                if batch:
                    results = self.embed_batch(batch)
                    for res in results:
                        storage_queue.put(res)
                embed_queue.task_done()
                break
            batch.append(task)
            if len(batch) >= self.batch_size:
                results = self.embed_batch(batch)
                for res in results:
                    storage_queue.put(res)
                batch = []
            embed_queue.task_done()
        # After receiving None, ensure any remaining batch is processed
        if batch:
            results = self.embed_batch(batch)
            for res in results:
                storage_queue.put(res)
