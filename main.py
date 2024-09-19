import sys
import os
import csv
import requests
import tempfile

# Add src folder to the path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from functools import partial
import glob
import h5py
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel
import torch
from pydub import AudioSegment
import librosa

class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_path, sample_rate, segment_length):
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.audio, _ = librosa.load(self.audio_path, sr=self.sample_rate)
        
    def __len__(self):
        return max(1, len(self.audio) // (self.sample_rate * self.segment_length))
    
    def __getitem__(self, idx):
        start = idx * self.sample_rate * self.segment_length
        end = start + self.sample_rate * self.segment_length
        audio_segment = self.audio[start:end]
        
        if len(audio_segment) < self.sample_rate * self.segment_length:
            audio_segment = librosa.util.fix_length(audio_segment, size=self.sample_rate * self.segment_length)
        
        return {'input_values': audio_segment, 'id': idx}

def custom_collate_fn(batch, processor):
    input_values = [item['input_values'] for item in batch]
    processed = processor(input_values, sampling_rate=processor.sampling_rate, return_tensors="pt", padding=True)
    ids = [item['id'] for item in batch]
    return {'input_values': processed.input_values, 'attention_mask': processed.attention_mask, 'id': ids}

def download_m4a(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download file from {url}")

def convert_m4a_to_wav(m4a_data):
    with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as temp_m4a:
        temp_m4a.write(m4a_data)
        temp_m4a_path = temp_m4a.name

    audio = AudioSegment.from_file(temp_m4a_path, format="m4a")
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        audio.export(temp_wav.name, format="wav")
        temp_wav_path = temp_wav.name

    os.unlink(temp_m4a_path)
    return temp_wav_path

def main():
    # Create necessary directories
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(current_dir, 'data', 'raw')
    csv_root = os.path.join(current_dir, 'data', 'csvs')
    embedding_root = os.path.join(current_dir, 'data', 'embeddings')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(csv_root, exist_ok=True)
    os.makedirs(embedding_root, exist_ok=True)

    # Set the device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model_name = "m-a-p/MERT-v1-95M"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.eval()
    model.to(device)

    processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    resample_rate = processor.sampling_rate

    # Sanitize model_name for filenames
    safe_model_name = model_name.replace('/', '_')

    # Print the current working directory for debugging
    print(f"Current working directory: {current_dir}")
    print(f"CSV root directory: {csv_root}")
    print(f"Save directory (audio files): {save_dir}")
    print(f"Embedding root directory: {embedding_root}")

    # Process CSV files
    csv_files = glob.glob(os.path.join(csv_root, "chunk_*.csv"))
    print(f"Found CSV files")
    for csv_file in csv_files:
        csv_file_name = os.path.basename(csv_file)
        print(f"\nProcessing CSV file: {csv_file}")

        # Read CSV and process each row
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in tqdm(csv_reader, desc="Processing tracks"):
                preview_url = row['previewUrl']
                track_id = row.get('id', 'unknown')  # Assuming there's an 'id' column, otherwise use 'unknown'

                try:
                    # Download and convert m4a to wav
                    m4a_data = download_m4a(preview_url)
                    temp_wav_path = convert_m4a_to_wav(m4a_data)

                    # Create a temporary dataset with this single file
                    temp_dataset = CustomAudioDataset(
                        audio_path=temp_wav_path,
                        sample_rate=resample_rate,
                        segment_length=10
                    )

                    my_collate_fn = partial(custom_collate_fn, processor=processor)
                    data_loader = torch.utils.data.DataLoader(
                        temp_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=False,
                        collate_fn=my_collate_fn,
                    )

                    with torch.inference_mode():
                        for input_audio in data_loader:
                            inputs = {
                                'input_values': input_audio['input_values'].to(device),
                                'attention_mask': input_audio.get('attention_mask', None)
                            }
                            if inputs['attention_mask'] is not None:
                                inputs['attention_mask'] = inputs['attention_mask'].to(device)

                            outputs = model(**inputs, output_hidden_states=True)

                            # Process outputs
                            all_layer_hidden_states = torch.stack(outputs.hidden_states).permute(1, 0, 2, 3)
                            time_reduced_hidden_states = all_layer_hidden_states.mean(-2)

                            # Save embeddings
                            embedding_dir = os.path.join(embedding_root, os.path.splitext(csv_file_name)[0])
                            os.makedirs(embedding_dir, exist_ok=True)
                            embedding_file = os.path.join(embedding_dir, f"audio_embeddings_{safe_model_name}_{track_id}.h5")
                            with h5py.File(embedding_file, 'w') as hf:
                                hf.create_dataset(str(track_id), data=time_reduced_hidden_states.cpu().numpy())

                    # Clean up temporary file
                    os.unlink(temp_wav_path)

                except Exception as e:
                    print(f"Error processing track ID {track_id}: {str(e)}")

        print(f"Completed processing CSV file: {csv_file}")

    print("All processing complete.")

if __name__ == "__main__":
    main()
