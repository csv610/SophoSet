import pandas as pd
import logging

from tqdm import tqdm
from llm_chat import LLMChat
from datasets import load_dataset

# Constants
MODEL_ID = "GBaker/MedQA-USMLE-4-options-hf"
SPLITS = ["train", "validation", "test"]
OUTPUT_FILE = "medqa_usmle_4_result.csv"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_subset(llm, split):
    try:
        dataset = load_dataset(MODEL_ID)
        split_data = dataset[split]
    except Exception as e:
        logger.error(f"Error loading dataset for {split} split: {str(e)}")
        return None

    data = []
    for i, row in tqdm(enumerate(split_data), total=len(split_data), desc=f"Processing {split}"):
        question = row['sent1']
        options = [row['ending0'], row['ending1'], row['ending2'], row['ending3']]
        
        try:
            llm_response = llm.get_answer(question, options)
            llm_answer = llm_response['answer']
        except Exception as e:
            logger.error(f"Error getting LLM answer for question {i+1} in {split} split: {str(e)}")
            llm_answer = None
        
        data.append({
            'id': f"{split}_{i+1}",
            'Question': question,
            'Options': options,
            'Answer': row['label'],
            'llama3.1': llm_answer
        })
    return pd.DataFrame(data)

def process_dataset():
    dataframes = {}
    llm = LLMChat("llama3.1")  # Assuming you have an LLMChat class defined
    for split in SPLITS:
        df = process_subset(llm, split)
        if df is not None:
            dataframes[split] = df
            logger.info(f"Loaded {len(df)} questions from {split} split")
        else:
            logger.warning(f"Failed to load data for {split} split")

    if dataframes:
        # Combine all DataFrames
        combined_df = pd.concat(dataframes.values(), ignore_index=True)
        
        # Write the combined DataFrame to a CSV file
        combined_df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Combined data written to {OUTPUT_FILE}")
        logger.info(f"Total questions: {len(combined_df)}")
    else:
        logger.error("No data was loaded. Unable to create output file.")

if __name__ == "__main__":
    process_dataset()

