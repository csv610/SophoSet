import math
import pandas as pd
import logging
import random
import argparse

from llm_chat import LLMChat
from tqdm import tqdm
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
def load_data():
    try:
        logging.info("Loading IMO geometry dataset")
        ds = load_dataset("theblackcat102/IMO-geometry")
        return ds['test']
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        return None

def process_subset(llm, nsamples=None):
    """Function to load and process the dataset, using random samples if specified."""
    dataset = load_data()
    if dataset is None:
        logging.error("Unable to load dataset. Please try again later.")
        return None
    
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))

    data = []
    for i in tqdm(indices, total=len(indices), desc="Processing questions"):
        row = dataset[i]
        question = row['question']
        logging.debug(f"Processing question {i + 1}")
        llm_response = llm.get_answer(question)
        llm_answer   = llm_response['answer']
        data.append({
            "Question Number": i + 1,
            "Question": question,
            "llama3.1": llm_answer
        })

    df = pd.DataFrame(data)
    return df

def process_dataset(nsamples=None):
    logging.info("Starting dataset processing: IMO Geometry")
    llm = LLMChat("llama3.1")
    df = process_subset(llm, nsamples)
    
    if df is not None:
        logging.info("Writing results to CSV file")
        df.to_csv("imo_geometry_result.csv", index=False)
        logging.info("Data has been written to imo_geometry_result.csv")
    else:
        logging.warning("No data to write to CSV")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IMO Geometry dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    args = parser.parse_args()

    process_dataset(nsamples=args.nsamples)
     
    
