import random
import argparse
import pandas as pd
import logging

from llm_chat import LLMChat
from datasets import load_dataset
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(split='train'):
    try:
        ds = load_dataset("xw27/scibench")
        logger.info(f"Successfully loaded {split} split of the dataset")
        return ds[split]
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def process_dataset(model_name, nsamples=None):
    dataset = load_data()
    if dataset is None:
        return

    data = []
    llm = LLMChat(model_name)
    logger.info(f"Processing dataset with {'all' if nsamples is None else nsamples} samples")

    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))

    for i in tqdm(indices, desc="Processing questions"):
        row = dataset[i]
        problem = row['problem_text']
        
        try:
            llm_response = llm.get_answer(problem)
            llm_answer = llm_response['answer']
        except Exception as e:
            logger.error(f"Error getting answer for question {i + 1}: {e}")
            llm_answer = "Error"
        
        data.append({
            'id': i + 1,
            'problem Text': problem, 
            'answer': row['answer_number'],
            'llama3.1': llm_answer
        })

    df = pd.DataFrame(data)
    if df.empty:
        logger.warning("No data to save, DataFrame is empty.")
        return

    filename = f'scibench_result_{model_name}.csv'  # Create filename
    df.to_csv(filename, index=False)
    logger.info("Dataset saved to scibench_dataset.csv")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MMLU dataset")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.1", required=True, help="Name of the model to use.")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    args = parser.parse_args()

    logger.info(f"Starting script with {args.nsamples if args.nsamples else 'all'} samples")
    process_dataset(model_name=args.model_name, nsamples=args.nsamples)
    logger.info("Script completed")
