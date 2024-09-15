import os
import pandas as pd
import logging
import random
import argparse

from tqdm import tqdm
from llm_chat import LLMChat
from datasets import load_dataset


# Constants
MODEL_ID = "GBaker/MedQA-USMLE-4-options"
SPLITS = ["train", "test"]


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_subset(llm, split, nsamples=None):
    try:
        dataset = load_dataset(MODEL_ID)
        dataset = dataset[split]
    except Exception as e:
        logger.error(f"Error loading dataset for {split} split: {str(e)}")
        return None
    
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))

    model_name = llm.get_model_name()

    data = []
    for i in tqdm(indices, desc=f"{split}", leave=False):
        row = dataset[i]
        question = row['question']
        options  = list(row['options'].values())
        answer    = row['answer_idx']
        
        try:
            llm_response = llm.get_answer(question, options)
            llm_answer = llm_response['answer']
        except Exception as e:
            logger.error(f"Error getting LLM answer for question {i+1} in {split} split: {str(e)}")
            llm_answer = None
        
        data.append({
            'id': f"{split}_{i+1}",
#           'Question': question,
#           'Options': options,
            'answer': answer,
            model_name: llm_answer
        })
    return pd.DataFrame(data)

def process_dataset(model_name, nsamples=None):
    dataframes = {}
    
    llm = LLMChat(model_name)

    for split in SPLITS:
        df = process_subset(llm, split, nsamples)
        if df is not None:
            dataframes[split] = df
            logger.info(f"Loaded {len(df)} questions from {split} split")
        else:
            logger.warning(f"Failed to load data for {split} split")

    if not os.path.exists('results'):
        os.makedirs('results')    

    if dataframes:
        # Combine all DataFrames
        combined_df = pd.concat(dataframes.values(), ignore_index=True)
        filename = f"medqa_usmle_4_option2_result_{model_name}.csv"
        # Write the combined DataFrame to a CSV file
        combined_df.to_csv(filename, index=False)
        logger.info(f"Combined data written to {OUTPUT_FILE}")
        logger.info(f"Total questions: {len(combined_df)}")
    else:
        logger.error("No data was loaded. Unable to create output file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MedQA USMLE-4 dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.1", help="Name of the model to use.")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Starting script with {args.nsamples if args.nsamples else 'all'} samples")
    process_dataset(model_name=args.model_name, nsamples=args.nsamples)
    logger.info("Script completed successfully")


