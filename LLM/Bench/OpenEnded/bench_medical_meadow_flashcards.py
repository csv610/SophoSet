import logging
import pandas as pd
import random
import argparse

from tqdm import tqdm
from llm_chat import LLMChat
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(split="train"):
    model_name = "medalpaca/medical_meadow_medical_flashcards"
    try:
        logger.info(f"Loading dataset: {model_name}")
        ds = load_dataset(model_name)
        logger.info(f"Dataset loaded successfully. Split: {split}")
        return ds[split]
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def process_subset(llm, nsamples=None):
    dataset = load_data()
    if dataset is None:
        logger.error("Unable to load dataset. Please try again later.")
        return None
    
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))

    data = []

    for i in tqdm(indices, desc="Processing flashcards"):
        row = dataset[i]
        question = row['input']

        try:
            llm_response = llm.get_answer(question)
            llm_answer = llm_response['answer']
        except Exception as e:
            logger.error(f"Error processing flashcard {i}: {str(e)}")
            llm_answer = "Error: Unable to process"

        data.append({
            "Question": question,
            "Answer": row['output'],
            "llama3.1": llm_answer
        })

    logger.info("Finished processing flashcards")
    return pd.DataFrame(data)

def process_dataset(model_name, nsamples=None):
    logger.info("Starting dataset processing")

    llm = LLMChat(model_name)

    df = process_subset(llm)
    if df is not None:
        filename = f"medical_meadow_flashcards_{model_name}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Results have been written to {csv_filename}")
    logger.info("Dataset processing completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Medical Meadow Flashcards dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.1", help="Model name to use for processing.")
    args = parser.parse_args()

    process_dataset(model_name=args.model_name, nsamples=args.nsamples)
