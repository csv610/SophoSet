import pandas as pd
import argparse
import random
import logging

from datasets import load_dataset
from llm_chat import LLMChat
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_subset(llm, split, nsamples = None):
    # Load the dataset inside the function
    dataset = load_dataset("automated-research-group/winogrande")[split]

    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))
    
    # Create a list to store the data
    data = []
    
    logger.info(f"Processing {len(indices)} samples from the {split} split")

    model_name = llm.get_model_name()
    
    for i in tqdm(indices, desc="Processing items"):
        row = dataset[i]
        question = row['request']

        try:
            llm_response = llm.get_answer(question)
            llm_answer = llm_response['answer']
            logger.debug(f"Generated response for question: {question[:50]}...")
        except Exception as e:
            logger.error(f"Error generating response for question: {question}")
            logger.error(f"Error details: {str(e)}")
            llm_answer = "Error"

        data.append({
            "id": f"{split}_{i+1}",
#           "question": question,
            "answer": row['response'],
            model_name: llm_answer
        })
    
    logger.info(f"Processed {len(data)} samples successfully")
    # Create a pandas DataFrame
    df = pd.DataFrame(data)
    return df

def process_dataset(nsamples=None):
    logger.info("Starting dataset processing")
    model_name = "llama3.1"
    llm = LLMChat(model_name)
    df = process_subset(llm, "validation", nsamples)

    # Save the DataFrame to a CSV file
    df.to_csv("winogrande_result.csv", index=False)
    logger.info("DataFrame saved to winogrande_result.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Winogrande dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Starting script with {args.nsamples if args.nsamples else 'all'} samples")
    process_dataset(nsamples=args.nsamples)
    logger.info("Script completed successfully")
