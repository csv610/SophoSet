import logging
import argparse
import random
import pandas as pd

from tqdm import tqdm
from llm_chat import LLMChat
from datasets import load_dataset

MODEL_ID = "lighteval/MATH"
SUBSETS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
SPLITS = ["train", "test"]

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(subset, split):
    try:
        ds = load_dataset(MODEL_ID, subset)
        logging.info(f"Successfully loaded dataset {subset} - {split}")
        return ds[split]
    except Exception as e:
        logging.error(f"Error loading dataset {subset} - {split}: {str(e)}")
        return None

def process_subset(llm, subset, split, nsamples=None):
    dataset = load_data(subset, split)
    if dataset is None:
        return pd.DataFrame()
    
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))
    
    subset_data = []
    for i in tqdm(indices, desc=f"Processing {subset} - {split}"):
        row = dataset[i]
        problem = row['problem']

        try:
            llm_response = llm.get_answer(problem)
            llm_answer   = llm_response['answer']
        except Exception as e:
            logging.error(f"Error getting LLM response for {subset} - {split}, index {i}: {str(e)}")
            llm_answer = "Error"

        subset_data.append({
            'id': f"{subset}_{split}_{i}",
            'problem': problem,
            'solution': row['solution'],
            'llama3.1': llm_answer
        })
    
    return pd.DataFrame(subset_data)

def process_dataset(nsamples=None):
    setup_logging()
    logging.info("Starting dataset processing")
    llm = LLMChat("llama3.1")
    frames = []

    for subset in SUBSETS:
        for split in SPLITS:
            logging.info(f"Processing subset: {subset}, split: {split}")
            newframe = process_subset(llm, subset, split, nsamples)
            frames.append(newframe)

    df = pd.concat(frames, ignore_index=True)

    df.to_csv('math_dataset.csv', index=False)
    logging.info("Data saved to math_dataset.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Math dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process per subset and split. If not provided, process all samples.")
    args = parser.parse_args()

    process_dataset(nsamples=args.nsamples)
    
    
