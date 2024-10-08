import pandas as pd
import logging
import random 
import argparse

from datasets import load_dataset
from llm_chat import LLMChat
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(split):
    try:
        ds = load_dataset("lavita/ChatDoctor-iCliniq")
        logger.info(f"Successfully loaded {split} split of the dataset")
        return ds[split]
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None
    
def process_subset(llm, split, nsamples=None):
    dataset = load_data(split)
    if dataset is None:
        logger.error("Failed to load the dataset. Please try again later.")
        return
    
    if nsamples is None or nsamples >= len(dataset):
        indices = range(len(dataset))
    else:
        indices = random.sample(range(len(dataset)), nsamples)
    
    results = []
    
    logger.info(f"Processing {len(indices)} samples")
    for i in tqdm(indices, desc="Processing samples"):
        row = dataset[i]
        question = row['input']
        answer = row['answer_icliniq']

        try:
            llm_response = llm.get_answer(question)
            llm_answer = llm_response['answer']
        except Exception as e:
            logger.error(f"Error processing question {i}: {str(e)}")
            llm_answer = "Error"
        
        results.append({
            'id': f"{split}_{i+1}",
            'Question': question,
            'Answer': answer,
            'llama3.1': llm_answer
        })

    # Create a pandas DataFrame
    df = pd.DataFrame(results)
    return df

def process_dataset(model_name, nsamples=None):
    logger.info("Starting process: Dataset: ChatDoctor iCliniq")

    llm = LLMChat(model_name)
    split = "train"

    df = process_subset(llm, split, nsamples)
    if df is empty:
        logger.error("The DataFrame is empty. No data to save.")
        return
    
    # Save the DataFrame to a CSV file
    filename = f'chatdoctor_icliniq_result_{model_name}.csv'
    df.to_csv(filename, index=False)
    logger.info(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ChatDoctor iCliniq dataset")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.1", required=True, help="Name of the model to use")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process")
    args = parser.parse_args()

    process_dataset(args.model_name, args.nsamples)

