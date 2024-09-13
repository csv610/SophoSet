import logging
import argparse
import random
import pandas as pd

from tqdm import tqdm
from llm_chat import LLMChat
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(subset, split):
    try:
        model_name = "Idavidrein/gpqa"
        hf_token = "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"
        ds = load_dataset(model_name, subset, use_auth_token=hf_token)
        logger.info(f"Successfully loaded dataset: {subset} - {split}")
        return ds[split]
    except Exception as e:
        logger.error(f"Error loading dataset {subset} - {split}: {str(e)}")
        return None

def process_subset(llm, subset, split, nsamples=None):
    """
    Process a specific subset and split of the GPQA dataset.
    
    This function loads the data, extracts relevant information,
    and formats it into a pandas DataFrame. It uses tqdm to show
    progress while processing the dataset.
    
    Args:
    llm (LLMChat): The LLM chat instance to use for generating answers
    subset (str): The subset of GPQA to process
    split (str): The split of the data to process
    
    Returns:
    DataFrame or None: A DataFrame containing the processed data, or None if loading fails
    """
    dataset = load_data(subset, split)
    if dataset is None:
        return None
    
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))
    
    data = []
    for i in tqdm(indices, desc=f"Processing {subset} - {split}"):
        row = dataset[i]
        question = row['Pre-Revision Question']

        try:
            llm_response = llm.get_answer(question)
            llm_answer = llm_response['answer']
        except Exception as e:
            logger.error(f"Error getting LLM response for question {i+1} in {subset} - {split}: {str(e)}")
            llm_answer = "Error"

        data.append({
            'id': f"{subset}_{split}_{i+1}",
            'question': question,
            'answer': row['Pre-Revision Correct Answer'],
            'llama3.1': llm_answer
        })
    
    logger.info(f"Completed processing {subset} - {split}")
    return pd.DataFrame(data)

def process_dataset(nsamples = None):
    """
    Main function to process all subsets and splits of the GPQA dataset.
    
    This function iterates through all combinations of subsets and splits,
    processes the data, combines it into a single DataFrame, and saves it to a CSV file.
    """
    logger.info("Starting GPQA dataset processing")
    
    subsets = ["gpqa_diamond", "gpqa_extended", "gpqa_main"]
    splits = ["train"]
    
    frames = []

    llm = LLMChat("llama3.1")
    
    for subset in subsets:
        for split in splits:
            df = process_subset(llm, subset, split, nsamples)
            if df is not None:
                frames.append(df)
                logger.info(f"Added {subset} - {split} to the combined dataset")
    
    combined_df = pd.concat(frames, ignore_index=True)
    
    output_file = "gpqa_result.csv"
    combined_df.to_csv(output_file, index=False)
    
    logger.info(f"Data written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GPQA dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    args = parser.parse_args()

    process_dataset(nsamples=args.nsamples)

