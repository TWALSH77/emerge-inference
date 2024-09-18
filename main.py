import sys
import os
import argparse
from functools import partial
import glob
import h5py
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel
from src.data.dataset import GenericAudioDataset, audio_collate_fn_segments
import torch
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate Audio Embeddings")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    parser.add_argument('--segment_length', type=int, default=10, help='Length of each audio segment in seconds')
    parser.add_argument('--overlap', type=int, default=1, help='Overlap between segments in seconds')
    parser.add_argument('--device', type=str, default=None, help='Device to use: cuda, mps, cpu')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    batch_size = args.batch_size
    num_workers = args.num_workers
    segment_length = args.segment_length
    overlap = args.overlap
    specified_device = args.device

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of workers: {num_workers}")
    logger.info(f"Segment length: {segment_length} seconds")
    logger.info(f"Overlap: {overlap} seconds")

    # Create necessary directories
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(current_dir, 'data', 'raw')
    csv_root = os.path.join(current_dir, 'data', 'csvs')
    embedding_root = os.path.join(current_dir, 'data', 'embeddings')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(csv_root, exist_ok=True)
    os.makedirs(embedding_root, exist_ok=True)

    logger.info(f"Save directory (audio files): {save_dir}")
    logger.info(f"CSV root directory: {csv_root}")
    logger.info(f"Embedding root directory: {embedding_root}")

    # Set the device
    if specified_device:
        device = torch.device(specified_device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load model and processor
    model_name = "m-a-p/MERT-v1-95M"
    logger.info(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device)

    processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    resample_rate = processor.sampling_rate
    logger.info(f"Processor sampling rate: {resample_rate}")

    # Sanitize model_name for filenames
    safe_model_name = model_name.replace('/', '_')

    # Process CSV files
    csv_files = glob.glob(os.path.join(csv_root, "*.csv"))
    logger.info(f"Found CSV files: {csv_files}")

    if not csv_files:
        logger.error("No CSV files found in the CSV root directory.")
        sys.exit(1)

    for csv_file in csv_files:
        csv_file_name = os.path.basename(csv_file)
        songs_root_path = os.path.join(save_dir, os.path.splitext(csv_file_name)[0])

        logger.info(f"\nProcessing CSV file: {csv_file}")

        # Initialize the dataset
        ds = GenericAudioDataset(
            csv_file=csv_file,
            root_path=save_dir,
            sample_rate=resample_rate,
            segment_length=segment_length,
            overlap=overlap,
        )
        logger.info(f"Dataset length: {len(ds)} segments")

        if len(ds) == 0:
            logger.warning("Dataset is empty. Please check if the audio files exist at the specified paths.")
            continue  # Skip to the next CSV file if dataset is empty

        # Initialize the DataLoader
        my_collate_fn = partial(audio_collate_fn_segments, processor=processor)
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            collate_fn=my_collate_fn,
        )
        logger.info("DataLoader initialized.")

        # Create embedding directory for the current CSV file
        embedding_dir = os.path.join(embedding_root, os.path.splitext(csv_file_name)[0])
        os.makedirs(embedding_dir, exist_ok=True)

        # Disable gradient calculations for inference
        with torch.inference_mode():
            for idx, input_audio in tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference"):
                if not input_audio:
                    logger.warning("Empty batch received. Skipping.")
                    continue

                ids = [int(x) for x in input_audio['id']]
                inputs = {
                    'input_values': input_audio['input_values'].to(device),
                    'attention_mask': input_audio.get('attention_mask', None)
                }

                # Move attention_mask to device if it exists
                if inputs['attention_mask'] is not None:
                    inputs['attention_mask'] = inputs['attention_mask'].to(device)

                # Debugging: log input shapes at debug level
                logger.debug(f"Input values shape: {inputs['input_values'].shape}")
                if inputs['attention_mask'] is not None:
                    logger.debug(f"Attention mask shape: {inputs['attention_mask'].shape}")

                # Forward pass through the model
                outputs = model(**inputs, output_hidden_states=True)

                # Process outputs
                all_layer_hidden_states = torch.stack(outputs.hidden_states).permute(1, 0, 2, 3)
                time_reduced_hidden_states = all_layer_hidden_states.mean(-2)

                # Aggregate embeddings per unique ID
                unique_ids = torch.unique(torch.tensor(ids))
                for unique_id in unique_ids:
                    indices = torch.where(torch.tensor(ids) == unique_id)[0]
                    time_reduced_hidden_states_ind = time_reduced_hidden_states[indices].mean(0, keepdim=True)

                    # Save embeddings
                    embedding_file = os.path.join(
                        embedding_dir,
                        f"audio_embeddings_{safe_model_name}_{unique_id.item()}.h5"
                    )
                    with h5py.File(embedding_file, 'w') as hf:
                        hf.create_dataset(str(unique_id.item()), data=time_reduced_hidden_states_ind.cpu().numpy())

        logger.info("Processing complete for CSV file: {}".format(csv_file_name))

    logger.info("All CSV files have been processed.")

if __name__ == "__main__":
    main()
