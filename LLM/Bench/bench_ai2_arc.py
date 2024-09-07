import logging
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from llm_chat import LLMChat

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_subset(llm, subset, split):
    try:
        ds = load_dataset("allenai/ai2_arc", subset)
        dataset = ds[split]
        logger.info(f"Successfully loaded dataset: {subset} - {split}")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

    data = []
    for i in tqdm(range(len(dataset)), desc=f"{subset} - {split}", leave=False):
        row = dataset[i]
        question = row['question']
        choices  = row['choices']['text']

        try:
            llm_response = llm.get_answer(question, choices)
            llm_answer   = llm_response['answer']
        except Exception as e:
            logger.error(f"Error getting answer from LLM for question {i}: {str(e)}")
            llm_answer = "Error"
        
        newdata = {
            'id': f"{subset}_{split}_{i}",
            'question': question,
            'choices': choices,
            'answer_key': row['answerKey'],
            'llama3.1': llm_answer
        }
        data.append(newdata)
        
    logger.info(f"Processed {len(data)} rows for {subset} - {split}")
    return pd.DataFrame(data)

def process_dataset():
    logger.info("Starting AI2 ARC dataset processing")

    subsets = ["ARC-Challenge", "ARC-Easy"]
    splits = ["train", "validation", "test"]

    frames = []

    llm = LLMChat("llama3.1")
    logger.info("Initialized LLMChat with model: llama3.1")

    for subset in subsets:
        for split in splits:
            df = process_subset(llm, subset, split)
    
            if df is not None:
                frames.append(df)
                logger.info(f"Processed {len(df)} rows for {subset} - {split}")
            else:
                logger.warning(f"Failed to load dataset for {subset} - {split}")

    # Combine all DataFrames
    final_df = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined all DataFrames. Total rows: {len(final_df)}")
    
    # Write to CSV
    csv_filename = 'ai2_arc_results.csv'
    final_df.to_csv(csv_filename, index=False)
    logger.info(f"All data written to {csv_filename}")

if __name__ == "__main__":
    process_dataset()

