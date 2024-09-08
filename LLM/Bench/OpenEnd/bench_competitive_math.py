import logging
import pandas as pd
import argparse
import random

from tqdm import tqdm
from llm_chat import LLMChat
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(split):
    model_name = "qwedsacf/competition_math"
    try:
        ds = load_dataset(model_name)
        ds = ds[split]
        logger.info(f"Successfully loaded {split} split from {model_name}")
        return ds
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def process_subset(llm, split, nsamples = None):
    # Load all dataset splits
    dataset = load_data(split)
    if dataset is None:
        logger.warning(f"Skipping {split} split due to loading error.")
        return pd.DataFrame()  # Return an empty DataFrame if loading fails
    
    data = []
    # If num_samples is provided and less than the dataset size, use random sampling
    if nsamples and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))
    
    for i in tqdm(indices, desc=f"Processing {split} split"):
        row = dataset[i]
        problem = row['problem']

        try:
            llm_response = llm.get_answer(problem)
            llm_answer = llm_response["answer"]
        except Exception as e:
            logger.error(f"Error processing problem {i}: {str(e)}")
            llm_answer = "Error"

        data.append({
            'id': f"{split}_{i+1}",
            'Problem': problem,
            'Solution': row['solution'],
            "llama3.1": llm_answer
        })
    return pd.DataFrame(data)

def process_dataset(nsamples = None):
    logger.info("Starting dataset processing")
    llm = LLMChat("llama3.1")

    # Process each split and store in a list
    dataframes = []

    SPLITS = ['train']

    for split in SPLITS:
        logger.info(f"Processing {split} split")
        df = process_subset(llm, split, nsamples)
        if df.empty:
            logger.warning(f"Skipping {split} split due to processing error.")
            continue
        dataframes.append(df)
        logger.info(f"Finished processing {split} split")

    # Combine all DataFrames
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        # Save the combined DataFrame to a CSV file
        combined_df.to_csv('competitive_math_result.csv', index=False)
        logger.info("Combined DataFrame saved to 'competitive_math_result.csv'")
    else:
        logger.warning("No data to combine and save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Competitive Math dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, 
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    args = parser.parse_args()

    process_dataset(nsamples=args.nsamples)
