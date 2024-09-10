import logging
import pandas as pd
import random
import argparse

from tqdm import tqdm
from llm_chat import LLMChat
from typing import List, Optional
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_subset(llm: LLMChat, subset: str, split: str, nsamples: Optional[int] = None) -> Optional[pd.DataFrame]:
    try:
        ds = load_dataset("allenai/ai2_arc", subset)
        dataset = ds[split]
        logger.info(f"Successfully loaded dataset: {subset} - {split}")
    except Exception as e:
        logger.error(f"Error loading dataset {subset} - {split}: {str(e)}")
        return None

    # If num_samples is provided and less than the dataset size, use random sampling
    if nsamples and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))

    model_name = llm.get_model_name()

    data = []
    for i in tqdm(indices, desc=f"{subset} - {split}", leave=False):
        row = dataset[i]
        question = row['question']
        choices: List[str] = row['choices']['text']

        try:
            llm_response = llm.get_answer(question, choices)
            llm_answer = llm_response['answer']
            logger.debug(f"Successfully processed question {i} for {subset} - {split}")
        except Exception as e:
            logger.error(f"Error getting answer from LLM for question {i} in {subset} - {split}: {str(e)}")
            llm_answer = "Error"
        
        newdata = {
            'id': f"{subset}_{split}_{i}",
            'question': question,
            'choices': choices,
            'answer_key': row['answerKey'],
            model_name: llm_answer
        }
        data.append(newdata)
        
    logger.info(f"Successfully processed {len(data)} rows for {subset} - {split}")
    return pd.DataFrame(data)

def process_dataset(nsamples: Optional[int] = None) -> None:
    logger.info("Starting AI2 ARC dataset processing")

    subsets = ["ARC-Challenge", "ARC-Easy"]
    splits = ["train", "validation", "test"]

    frames = []

    model_name = "llama3.1"

    llm = LLMChat(model_name)
    logger.info("Initialized LLMChat with model: f"{model_name}")

    for subset in subsets:
        for split in splits:
            df = process_subset(llm, subset, split, nsamples=nsamples)
    
            if df is not None:
                frames.append(df)
            else:
                logger.warning(f"Failed to load dataset for {subset} - {split}")

    # Combine all DataFrames
    final_df = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined all DataFrames. Total rows: {len(final_df)}")
    
    # Write to CSV
    csv_filename = 'ai2_arc_results.csv'
    final_df.to_csv(csv_filename, index=False)
    logger.info(f"All data written to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AI2 ARC dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, 
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    args = parser.parse_args()

    process_dataset(nsamples=args.nsamples)
