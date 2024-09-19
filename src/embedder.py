# src/embedder.py

import os
import h5py
import torch
import torchaudio
import logging
from transformers import AutoFeatureExtractor, AutoModel
from queue import Queue
from typing import List, Tuple

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str, device: torch.device, processor: AutoFeatureExtractor, embedding_root: str, batch_size: int = 16):
        """
        Initializes the Embedder with the specified model and device.

        Args:
            model_name (str): The name of the pretrained model.
            device (torch.device): The device to run the model on.
            processor (AutoFeatureExtractor): The feature extractor for preprocessing.
            embedding_root (str): Directory to save embeddings.
            batch_size (int): Number of samples per batch.
        """
        self.model_name = model_name
        self.device = device
        self.processor = processor
        self.embedding_root = embedding_root
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    def embed_batch(self, batch: List[Tuple[int, str]]):
        """
        Embeds a batch of WAV files and saves the embeddings.

        Args:
            batch (List[Tuple[int, str]]): List of tuples containing song IDs and WAV file paths.

        Returns:
            List[Tuple[int, str]]: List of tuples containing song IDs and embedding file paths.
        """
        try:
            song_ids, wav_paths = zip(*batch)
            waveforms = []
            for wav_path in wav_paths:
                waveform, sample_rate = torchaudio.load(wav_path)
                waveforms.append(waveform.squeeze().numpy())

            inputs = self.processor(waveforms, sampling_rate=self.processor.sampling_rate, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            with torch.inference_mode():
                outputs = self.model(**inputs, output_hidden_states=True)
            # Assuming the model returns hidden states
            all_layer_hidden_states = torch.stack(outputs.hidden_states).permute(1, 0, 2, 3)
            time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
            # Averaging across time and layers
            embeddings = time_reduced_hidden_states.mean(dim=1).cpu().numpy()

            embedding_files = []
            for song_id, embedding, wav_path in zip(song_ids, embeddings, wav_paths):
                embedding_dir = os.path.join(self.embedding_root, f"audio_embeddings_{self.model_name.replace('/', '_')}")
                os.makedirs(embedding_dir, exist_ok=True)
                embedding_file = os.path.join(embedding_dir, f"audio_embeddings_{self.model_name.replace('/', '_')}_{song_id}.h5")
                with h5py.File(embedding_file, 'w') as hf:
                    hf.create_dataset(str(song_id), data=embedding)
                embedding_files.append((song_id, embedding_file))
                logger.info(f"Embedded and saved {wav_path} to {embedding_file}")
            return embedding_files
        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            return []

    def embed_worker(self, embed_queue: Queue, storage_queue: Queue):
        """
        Worker function to embed WAV files in batches.

        Args:
            embed_queue (Queue): Queue containing embedding tasks.
            storage_queue (Queue): Queue to enqueue embeddings for storage.
        """
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
