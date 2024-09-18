import os
import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import logging
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_audio_file(args):
    """
    Function to process a single audio file and generate segments.
    This function is designed to be used with multiprocessing.Pool.
    
    Args:
        args (tuple): Contains (row, root_dir, sample_rate, segment_length, overlap)
    
    Returns:
        list: List of segment dictionaries for the audio file.
    """
    row, root_dir, sample_rate, segment_length, overlap = args
    segments = []
    audio_path = os.path.join(root_dir, os.path.splitext(os.path.basename(row['csv_file']))[0], row['filename'])
    
    try:
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
        audio_length = waveform.shape[1] / sample_rate

        num_segments = int((audio_length - segment_length) // (segment_length - overlap) + 1)
        
        if num_segments > 0:
            for i in range(num_segments):
                start = int((segment_length - overlap) * i * sample_rate)
                end = start + int(segment_length * sample_rate)
                segments.append({
                    'id': row['id'],
                    'audio_path': audio_path,
                    'start': start,
                    'end': end,
                })
        else:
            logger.warning(f"No segments generated for file {audio_path}")
    except Exception as e:
        logger.error(f"Error processing audio file {audio_path}: {e}")
    
    return segments

class GenericAudioDataset(Dataset):
    """Audio dataset that handles loading and segmenting audio files."""

    def __init__(
        self,
        csv_file: str,
        root_path: str,
        transform=None,
        sample_rate: int = 16000,
        normalize: bool = False,
        segment_length: int = 10,
        overlap: int = 1,
        max_segments: int = None,  # Optional: Limit segments for testing
    ):
        """
        Initializes the dataset by reading the CSV file and preparing audio segments.

        Args:
            csv_file (str): Path to the CSV file containing audio metadata.
            root_path (str): Root directory where audio files are stored.
            transform: Optional transform to be applied on a sample.
            sample_rate (int, optional): Desired sample rate for audio. Defaults to 16000.
            normalize (bool, optional): Whether to normalize audio waveforms. Defaults to False.
            segment_length (int, optional): Length of each audio segment in seconds. Defaults to 10.
            overlap (int, optional): Overlap between consecutive segments in seconds. Defaults to 1.
            max_segments (int, optional): Maximum number of segments to generate. Defaults to None.
        """
        start_time = time.time()
        self.audio_metadata = pd.read_csv(csv_file)
        self.csv_file = os.path.splitext(os.path.basename(csv_file))[0]
        self.root_dir = root_path
        self.transform = transform
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.segment_length = segment_length
        self.overlap = overlap
        self.max_segments = max_segments

        logger.info(f"Loading CSV file: {csv_file}")
        
        # Drop duplicates and reset index
        self.audio_metadata = self.audio_metadata.drop_duplicates(subset=['id']).reset_index(drop=True)

        # Add a column to indicate file existence
        self.audio_metadata['file_exists'] = self.audio_metadata['filename'].apply(self._check_file_exists)
        self.audio_metadata = self.audio_metadata[self.audio_metadata['file_exists']].reset_index(drop=True)

        logger.info(f"Number of audio files found after filtering: {len(self.audio_metadata)}")
        logger.info(f"Root directory: {self.root_dir}")
        logger.info(f"CSV file name: {self.csv_file}")

        # Prepare arguments for multiprocessing
        process_args = [
            (
                row,
                self.root_dir,
                self.sample_rate,
                self.segment_length,
                self.overlap
            )
            for _, row in self.audio_metadata.iterrows()
        ]

        # Use multiprocessing Pool to parallelize segment generation
        num_processes = min(cpu_count(), 8)  # Limit to 8 processes to prevent overloading
        logger.info(f"Starting parallel segment generation using {num_processes} processes.")
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_audio_file, process_args)

        # Flatten the list of lists
        self.segments = [segment for sublist in results for segment in sublist]

        if self.max_segments:
            self.segments = self.segments[:self.max_segments]
            logger.info(f"Limiting to first {self.max_segments} segments.")

        end_time = time.time()
        logger.info(f"Dataset initialization completed in {end_time - start_time:.2f} seconds.")
        logger.info(f"Total segments generated: {len(self.segments)}")

    def _check_file_exists(self, filename: str) -> bool:
        """
        Checks if the audio file exists.

        Args:
            filename (str): Name of the audio file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        audio_path = os.path.join(self.root_dir, self.csv_file, filename)
        exists = os.path.exists(audio_path)
        if not exists:
            logger.warning(f"File does not exist: {audio_path}")
        return exists

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        """
        Retrieves the audio segment and its corresponding ID.

        Args:
            idx (int): Index of the segment.

        Returns:
            dict: Dictionary containing 'id' and 'audio' tensors.
        """
        segment = self.segments[idx]
        audio_path = segment['audio_path']
        start = segment['start']
        num_frames = segment['end'] - segment['start']
        try:
            waveform, sr = torchaudio.load(audio_path, frame_offset=start, num_frames=num_frames)
            waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            if self.normalize:
                waveform = (waveform - waveform.mean()) / waveform.std()
            if self.transform:
                waveform = self.transform(waveform)
            return {'id': segment['id'], 'audio': waveform}
        except Exception as e:
            logger.error(f"Error loading audio segment {audio_path}: {e}")
            return None

def audio_collate_fn_segments(batch, processor):
    """
    Collate function to combine a list of samples into a batch.

    Args:
        batch (list): List of samples.
        processor: Feature extractor processor.

    Returns:
        dict: Dictionary containing processed input tensors and IDs.
    """
    # Filter out any None samples
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}

    ids = [item['id'] for item in batch]
    audios = [item['audio'].squeeze().numpy() for item in batch]

    # Process audio data using the processor
    inputs = processor(
        audios,
        sampling_rate=processor.sampling_rate,
        padding=True,
        return_tensors="pt"
    )

    inputs['id'] = ids
    return inputs
