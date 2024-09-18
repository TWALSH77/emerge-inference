import sys
import os

# Add src folder to the path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from functools import partial
import glob
import h5py
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel
from src.data.dataset import GenericAudioDataset, audio_collate_fn_segments
import torch

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
    csv_files = glob.glob(os.path.join(csv_root, "*.csv"))
    print(f"Found CSV files: {csv_files}")
    for csv_file in csv_files:
        csv_file_name = os.path.basename(csv_file)
        songs_root_path = os.path.join(save_dir, os.path.splitext(csv_file_name)[0])

        print(f"\nProcessing CSV file: {csv_file}")

        ds = GenericAudioDataset(
            csv_file=csv_file,
            root_path=save_dir,
            sample_rate=resample_rate,
            segment_length=10,
            overlap=1,
        )
        print(f"Dataset length: {len(ds)}")

        if len(ds) == 0:
            print("Dataset is empty. Please check if the audio files exist at the specified paths.")
            continue  # Skip to the next CSV file if dataset is empty

        my_collate_fn = partial(audio_collate_fn_segments, processor=processor)
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=my_collate_fn,
        )

        with torch.inference_mode():
            for idx, input_audio in tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference"):
                ids = [int(x) for x in input_audio['id']]
                inputs = {
                    'input_values': input_audio['input_values'].to(device),
                    'attention_mask': input_audio.get('attention_mask', None)
                }
                if inputs['attention_mask'] is not None:
                    inputs['attention_mask'] = inputs['attention_mask'].to(device)

                # Debugging: print input shapes
                print(f"Input values shape: {inputs['input_values'].shape}")
                if inputs['attention_mask'] is not None:
                    print(f"Attention mask shape: {inputs['attention_mask'].shape}")

                outputs = model(**inputs, output_hidden_states=True)

                # Process outputs
                all_layer_hidden_states = torch.stack(outputs.hidden_states).permute(1, 0, 2, 3)
                time_reduced_hidden_states = all_layer_hidden_states.mean(-2)

                for id in torch.unique(torch.tensor(ids)):
                    indices = torch.where(torch.tensor(ids) == id)[0]
                    time_reduced_hidden_states_ind = time_reduced_hidden_states[indices].mean(0, keepdim=True)

                    # Save embeddings
                    embedding_dir = os.path.join(embedding_root, os.path.splitext(csv_file_name)[0])
                    os.makedirs(embedding_dir, exist_ok=True)
                    embedding_file = os.path.join(embedding_dir, f"audio_embeddings_{safe_model_name}_{id.item()}.h5")
                    with h5py.File(embedding_file, 'w') as hf:
                        hf.create_dataset(str(id.item()), data=time_reduced_hidden_states_ind.cpu().numpy())

        print("Processing complete.")

if __name__ == "__main__":
    main()
