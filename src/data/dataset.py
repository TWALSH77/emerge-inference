import os
import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import logging

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenericAudioDataset(Dataset):
    """Audio dataset."""

    def __init__(
        self,
        csv_file=None,
        root_path=None,
        transform=None,
        sample_rate=16000,
        normalize: bool = False,
        segment_length: int = 10,
        overlap: int = 1,
    ):
        self.audio_metadata = pd.read_csv(csv_file)
        self.csv_file_name = os.path.splitext(os.path.basename(csv_file))[0]
        self.root_dir = root_path
        self.transform = transform
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.segment_length = segment_length
        self.overlap = overlap

        # Drop duplicates and reset index
        self.audio_metadata = self.audio_metadata.drop_duplicates(subset=['id'])
        self.audio_metadata = self.audio_metadata.reset_index(drop=True)

        # Filter audio_metadata based on the existence of the audio files
        def check_file_exists(x):
            audio_path = os.path.join(self.root_dir, self.csv_file_name, x)
            exists = os.path.exists(audio_path)
            print(f"Checking if file exists: {audio_path} -> {exists}")
            return exists

        self.audio_metadata['file_exists'] = self.audio_metadata['filename'].apply(check_file_exists)
        self.audio_metadata = self.audio_metadata[self.audio_metadata['file_exists']]

        print(f"Number of audio files found after filtering: {len(self.audio_metadata)}")
        print(f"Root directory: {self.root_dir}")
        print(f"CSV file name: {self.csv_file_name}")

        # Generate segments
        self.segments = []
        for idx, row in self.audio_metadata.iterrows():
            audio_path = os.path.join(self.root_dir, self.csv_file_name, row['filename'])
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
                if sample_rate != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                    waveform = resampler(waveform)
                audio_length = waveform.shape[1] / self.sample_rate
                print(f"Audio length for file {audio_path}: {audio_length} seconds")
                num_segments = int((audio_length - self.segment_length) // (self.segment_length - self.overlap) + 1)
                print(f"Number of segments for file {audio_path}: {num_segments}")
                if num_segments > 0:
                    for i in range(num_segments):
                        start = int((self.segment_length - self.overlap) * i * self.sample_rate)
                        end = start + int(self.segment_length * self.sample_rate)
                        self.segments.append({
                            'id': row['id'],
                            'audio_path': audio_path,
                            'start': start,
                            'end': end,
                        })
                else:
                    print(f"No segments generated for file {audio_path}")
            except Exception as e:
                print(f"Error loading audio file {audio_path}: {e}")
                continue

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        audio_path = segment['audio_path']
        start = segment['start']
        num_frames = segment['end'] - segment['start']
        try:
            waveform, sample_rate = torchaudio.load(audio_path, frame_offset=start, num_frames=num_frames)
            waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            if self.normalize:
                waveform = (waveform - waveform.mean()) / waveform.std()
            data = {'id': segment['id'], 'audio': waveform}
            return data
        except Exception as e:
            print(f"Error loading audio segment {audio_path}: {e}")
            return None

def audio_collate_fn_segments(batch, processor):
    batch = [b for b in batch if b is not None]
    ids = [item['id'] for item in batch]
    audios = [item['audio'].squeeze().numpy() for item in batch]
    # Debugging: print audio shapes
    print(f"Audio shapes: {[audio.shape for audio in audios]}")
    inputs = processor(audios, sampling_rate=processor.sampling_rate, padding=True, return_tensors="pt")
    # Debugging: print input shapes
    print(f"Input values shape after processor: {inputs['input_values'].shape}")
    inputs['id'] = ids
    return inputs
