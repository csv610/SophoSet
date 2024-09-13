import pandas as pd
import random
import argparse
import logging

from tqdm import tqdm
from llm_chat import LLMChat
from datasets import load_dataset

# Define SPLITS constant
SPLITS = ["test", "testmini"]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(split):
    model_name = "qintongli/GSM-Plus"
    try:
        logger.info(f"Loading dataset '{model_name}' for split '{split}'")
        ds = load_dataset(model_name)
        if split not in ds:
            raise ValueError(f"Split '{split}' not found in the dataset.")
        logger.info(f"Successfully loaded {len(ds[split])} samples for split '{split}'")
        return ds[split]
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def process_subset(llm, split, nsamples=None):
    dataset = load_data(split)
    if dataset is None:
        logger.warning(f"No data available for split '{split}'. Skipping.")
        return pd.DataFrame()
    
    total_samples = len(dataset)
    if nsamples is None or nsamples >= total_samples:
        nsamples = total_samples
        indices = range(total_samples)
    else:
        indices = random.sample(range(total_samples), nsamples)
    
    logger.info(f"Processing {nsamples} samples for split '{split}'")
    subset_data = []
    for i in tqdm(indices, desc=f"Processing {split}"):
        row = dataset[i]
        question = row['question']
        answer = row['answer']
        
        try:
            llm_response = llm.get_answer(question)
            llm_answer = llm_response['answer']
        except Exception as e:
            logger.error(f"Error getting LLM response for question {i+1} in {split}: {str(e)}")
            llm_answer = "Error"

        subset_data.append({
            'id': f"{split}_{i+1}",
            'question': question,
            'correct_answer': answer,
            'llm_answer': llm_answer
        })
    
    logger.info(f"Completed processing {nsamples} samples for split '{split}'")
    return pd.DataFrame(subset_data)

def process_dataset(model_name, nsamples=None):
    frames = []
    
    logger.info(f"Initializing LLMChat with model '{model_name}'")

    llm = LLMChat(model_name)

    for split in SPLITS:
        logger.info(f"Starting processing for split '{split}'")
        subset_data = process_subset(llm, split, nsamples)
        frames.append(subset_data)
        logger.info(f"Finished processing split '{split}'")

    result_df = pd.concat(frames, ignore_index=True)
    if result_df.empty:
        logger.warning("No data to save. The DataFrame is empty.")
        return result_df
    
    output_file = f'gsm_plus_result_{model_name}.csv'
    logger.info(f"Saving results to '{output_file}'")
    result_df.to_csv(output_file, index=False)
    logger.info(f"Results saved successfully. Total samples processed: {len(result_df)}")
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GSM-Plus benchmark")
    parser.add_argument("-n","--nsamples", type=int, default=None, 
                        help="Number of samples to process per split. If not specified, all samples will be processed.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.1", 
                        help="Name of the model to use.")
    args = parser.parse_args()
    process_dataset(model_name=args.model_name, nsamples=args.nsamples)

