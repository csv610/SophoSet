
import random
import logging
import argparse
import pandas as pd

from tqdm import tqdm
from llm_chat import LLMChat
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(split='train'):
    try:
        ds = load_dataset("meta-math/MetaMathQA")
        if split not in ds:
            raise ValueError(f"Invalid split: {split}. Available splits are: {', '.join(ds.keys())}")
        logger.info(f"Successfully loaded {split} split of MetaMathQA dataset")
        return ds[split]
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def process_subset(llm, split='train', nsamples=None):
    dataset = load_data(split)
    if dataset is None:
        return None
    
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
        logger.info(f"Processing {nsamples} random samples from {split} split")
    else:
        indices = range(len(dataset))
        logger.info(f"Processing all {len(dataset)} samples from {split} split")

    subset_data = []
    for i in tqdm(indices, desc="Processing ..."):
        row = dataset[i]
        llm_response = llm.get_answer(row['query'], row['response'])
        llm_answer = llm_response['answer']
        subset_data.append({
            'id': f"{split}_{i + 1}",
            'Query': row['query'],
            'Response': row['response'],
            'llama3.1': llm_answer
        })
    logger.info(f"Processed {len(subset_data)} samples")
    return pd.DataFrame(subset_data)

def process_dataset(nsamples=None):
    logger.info("Starting MetaMathQA dataset processing")
    llm = LLMChat("llama3.1")
    logger.info("Initialized LLMChat with llama3.1 model")

    # Process the dataset
    df = process_subset(llm, nsamples=nsamples)
    if df is None:
        logger.error("Failed to process dataset")
        return

    # Write DataFrame to CSV
    csv_filename = "metamathqa_result.csv"
    df.to_csv(csv_filename, index=False)
    logger.info(f"Results written to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MetaMathQA dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    args = parser.parse_args()

    logger.info(f"Starting script with nsamples={args.nsamples}")
    process_dataset(nsamples=args.nsamples)
    logger.info("Script completed")
