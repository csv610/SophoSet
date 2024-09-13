import json
import logging
import argparse
import random
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from llm_chat import LLMChat

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(split: str = "train"):
    """Load the dataset."""
    try:
        ds = load_dataset("fhirfly/medicalquestions")
        logger.info(f"Successfully loaded {split} dataset")
        return ds[split]
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def process_subset(llm, split='train', nsamples=None):
    logger.info(f"Processing subset: split={split}, nsamples={nsamples}")
    
    dataset = load_data(split)
    if dataset is None:
        return
    
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))

    data = []
    for i in tqdm(indices, desc="Processing questions"):
        row = dataset[i]
        question = row['text']
        response = llm.get_answer(question)

        user_question = "What is the treatment?"  # Example user question
        prompt = f"Based on the Context of {question} Answer the User Question: {user_question}"
        try:
            llm_response = llm.get_answer(prompt)
            llm_answer = llm_response['answer']
        except Exception as e:
            logger.error(f"Error processing question {i}: {str(e)}")
            llm_answer = "Error: Unable to process this question"

        data.append({
            "Question": question,
            "llama3.1": llm_answer
        })

    df = pd.DataFrame(data)
    return df

def process_dataset(model_name, nsamples=None):
    logger.info(f"Processing Medical Questions dataset with {nsamples if nsamples else 'all'} samples")
    
    llm = LLMChat(model_name)

    df = process_subset(llm, nsamples=nsamples)

    if df is not None:
        filename = f"medicalquestions_result_{model_name}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")
    else:
        logger.warning("No data to save")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Medical Questions dataset")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.1", required=True, help="Name of the model to use.")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    args = parser.parse_args()

    process_dataset(model_name=args.model_name, nsamples=args.nsamples)
