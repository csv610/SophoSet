import math
from datasets import load_dataset
import pandas as pd
from llm_chat import LLMChat
from tqdm import tqdm
import logging

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

def process_subset(llm):
    """Function to load and process the entire dataset."""
    dataset = load_data()
    if dataset is None:
        logging.error("Unable to load dataset. Please try again later.")
        return None

    data = []
    for i, row in tqdm(enumerate(dataset), total=len(dataset), desc="Processing questions"):
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

def process_dataset():
    logging.info("Starting dataset processing: IMO Geometry")
    llm = LLMChat("llama3.1")
    df = process_subset(llm)
    
    if df is not None:
        logging.info("Writing results to CSV file")
        df.to_csv("imo_geometry_result.csv", index=False)
        logging.info("Data has been written to imo_geometry_result.csv")
    else:
        logging.warning("No data to write to CSV")

if __name__ == "__main__":
    process_dataset()
     
    
