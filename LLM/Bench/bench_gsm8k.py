import logging
from datasets import load_dataset
import re
import pandas as pd
from llm_chat import LLMChat
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(subset, split):
    try:
        model_name = "openai/gsm8k"
        ds = load_dataset(model_name, subset)
        logger.info(f"Successfully loaded dataset: {subset}/{split}")
        return ds[split]
    except Exception as e:
        logger.error(f"Error loading dataset {subset}/{split}: {str(e)}")
        return None

def split_solution_answer(text):
    parts = text.split('####')
    if len(parts) == 2:
        solution = parts[0].strip()
        answer = parts[1].strip()
        return answer, solution
    return text, ""

def process_subset(llm, subset, split):
    dataset = load_data(subset, split)
    
    if dataset is None:
        logger.warning(f"Unable to load dataset for {subset}/{split}. Skipping to next.")
        return None

    data = []
    for i in tqdm(range(len(dataset)), desc=f"Processing {subset}/{split}"):
        row = dataset[i]
        question = row['question']

        try:
            llm_response = llm.get_answer(question)
            llm_answer = llm_response['answer']
        except Exception as e:
            logger.error(f"Error getting answer for question {i} in {subset}/{split}: {str(e)}")
            llm_answer = "Error"

        raw_answer = row['answer']
        answer, solution = split_solution_answer(raw_answer)
        data.append({
            'id': f"{subset}_{split}_{i}",
            'question': question,
            'answer': answer,
            'llama3.1': llm_answer
        })
    
    logger.info(f"Completed processing {subset}/{split}")
    return pd.DataFrame(data)

def process_dataset():
    logger.info("Starting dataset processing: GSM8K")
    
    subsets = ["main", "socratic"]
    splits = ["train", "test"]

    llm = LLMChat("llama3.1")
    logger.info("Initialized LLMChat with model: llama3.1")

    frames = []
    for subset in subsets:
        for split in splits:
            df = process_subset(llm, subset, split)
            if df is not None:
                frames.append(df)

    # Combine all dataframes
    combined_df = pd.concat(frames, ignore_index=True)

    # Write to CSV
    output_file = "gsm8k_result.csv"
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Combined data written to {output_file}")

if __name__ == "__main__":
    process_dataset()
