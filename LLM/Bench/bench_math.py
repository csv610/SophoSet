import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from llm_chat import LLMChat
import logging

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

def process_subset(llm, subset, split):
    dataset = load_data(subset, split)
    if dataset is None:
        return pd.DataFrame()

    subset_data = []
    for i in tqdm(range(len(dataset)), desc=f"Processing {subset} - {split}"):
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

def process_dataset():
    setup_logging()
    logging.info("Starting dataset processing")
    llm = LLMChat("llama3.1")
    frames = []

    for subset in SUBSETS:
        for split in SPLITS:
            logging.info(f"Processing subset: {subset}, split: {split}")
            newframe = process_subset(llm, subset, split)
            frames.append(newframe)

    df = pd.concat(frames, ignore_index=True)

    df.to_csv('math_dataset.csv', index=False)
    logging.info("Data saved to math_dataset.csv")

if __name__ == "__main__":
    process_dataset()
    
    
