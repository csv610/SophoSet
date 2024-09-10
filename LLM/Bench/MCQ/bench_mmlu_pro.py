import logging
import string
import pandas as pd
import random
import argparse

from llm_chat import LLMChat
from tqdm import tqdm
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_subset(llm, split, nsamples=None):
    logger.info(f"Processing subset: {split}")
    
    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
        logger.info(f"Dataset loaded successfully for split '{split}'")
    except Exception as e:
        logger.error(f"Error loading dataset for split '{split}': {str(e)}")
        return None
    
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
        logger.info(f"Randomly sampled {nsamples} questions from {split} split")
    else:
        indices = range(len(dataset))
        logger.info(f"Processing all {len(dataset)} questions from {split} split")

    model_name = llm.get_model_name()

    data = []
    for i in tqdm(indices, total=len(indices), desc=f"Processing questions ({split})"):
        row = dataset[i]
        question = row['question']
        choices = row['options']
        answer = row['answer']

        try:
            llm_response = llm.get_answer(question, choices)
            llm_answer = llm_response['answer']
        except Exception as e:
            logger.error(f"Error getting answer for question {i}: {str(e)}")
            llm_answer = "Error"

        data.append({
            'id':f"{split}_{i+1}",
#           'Question': question,
#           'Choices': choices,
            'answer': answer,
            model_name: llm_answer
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Processed {len(df)} questions for {split} split")
    return df

def process_dataset(nsamples = None):
    logger.info("Starting dataset processing: MMLU Pro")
    
    model_name = "llama3.1"
    llm = LLMChat(model_name,temperature=0.0)
    logger.info(f"Initialized LLMChat with model: {model_name}")
    
    # List of available splits
    splits = ["test", "validation"]
    
    # Collect all DataFrames
    dataframes = []
    for split in splits:
        df = process_subset(llm, split, nsamples)
        if df is not None:
            dataframes.append(df)
    
    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined DataFrame created with {len(combined_df)} total questions")
    
    # Write the combined DataFrame to a CSV file
    combined_df.to_csv("mmlu_pro_result.csv", index=False)
    logger.info("Combined DataFrame written to mmlu_pro_result.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MMLU Pro dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None,
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    args = parser.parse_args()
    
    process_dataset(nsamples=args.nsamples)
