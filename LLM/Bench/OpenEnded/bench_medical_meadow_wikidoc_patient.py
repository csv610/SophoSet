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

def load_data(split):
    try:
        ds = load_dataset("medalpaca/medical_meadow_wikidoc_patient_information")
        logger.info(f"Successfully loaded {split} dataset")
        return ds[split]
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def process_subset(llm, split="train", nsamples=None):
    dataset = load_data(split)
    
    if dataset is None:
        logger.error("Unable to load the dataset. Please try again later.")
        return None

    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
        logger.info(f"Processing {nsamples} random samples")
    else:
        indices = range(len(dataset))
        logger.info(f"Processing all {len(dataset)} samples")

    data = []

    for i in tqdm(indices, desc="Processing dataset"):
        row = dataset[i]
        question = row['input']

        try:
            llm_response = llm.get_answer(question)
            llm_answer = llm_response['answer']
        except Exception as e:
            logger.error(f"Error getting LLM answer for question {i + 1}: {str(e)}")
            llm_answer = "Error: Unable to get answer"

        data.append({
            'id': f"{split}_{i + 1}",
            'Question': question,
            'Answer': row['output'],
            'llama3.1': llm_answer
        })

    df = pd.DataFrame(data)
    logger.info(f"Processed {len(data)} samples")
    return df

def process_dataset(model_name, nsamples=None):
    logger.info("Dataset: Medical Meadow Wikidoc Patient")

    llm = LLMChat(model_name)

    df = process_subset(llm, nsamples=nsamples)
    
    if df is not None:
        filename = f'medical_meadow_wikidoc_patient_result_{model_name}.csv'
        df.to_csv(filename, index=False)
        logger.info("DataFrame saved to medical_meadow_wikidoc_patient_result.csv")
    else:
        logger.error("Failed to process the dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Medical Meadow Wikidoc Patient dataset")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.1", required=True, help="Name of the model to use.")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    args = parser.parse_args()

    process_dataset(args.model_name, nsamples=args.nsamples)
