import re
import os
import ast
import random
import logging
import argparse
import pandas as pd

from tqdm import tqdm
from llm_chat import LLMChat
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_subset(llm, split = "train", nsamples=None):
    try:
        ds = load_dataset("medalpaca/medical_meadow_medqa")
        dataset = ds[split]
        logger.info(f"Dataset loaded successfully. Split: {split}, Size: {len(dataset)}")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return
    
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))

    model_name = llm.get_model_name()
    
    data = []
    for i in tqdm(indices, desc="Processing questions"):
        row = dataset[i]

        question = row['input']

        answer = row['output']
        match = re.search(r"\?\s*(\{.*\})", question)
        if match:
            question = question[:match.start()] + "?"
            choices_dict_str = match.group(1)
            choices_dict = ast.literal_eval(choices_dict_str)
            choices = list(choices_dict.values())
        else:
            choices = []

        answer = answer.strip()[0]

        try:
            llm_response = llm.get_answer(question, choices)
            llm_answer = llm_response['answer']
            logger.debug(f"LLM answer obtained for question {i+1}")
        except Exception as e:
            logger.error(f"Error getting LLM answer for question {i + 1}: {str(e)}")
            llm_answer = "Error"

        data.append({
            'id': f"{split}_{i+1}",
#           'Question': question_text,
#           'Choices': choices,
            'answer': answer,
            model_name: llm_answer
        })

    df = pd.DataFrame(data)
    logger.info(f"Processed {len(df)} questions for {split} split")
    return df

def process_dataset(model_name, nsamples=None):
    logger.info("Starting dataset processing: medical meadow medqa")

    llm = LLMChat(model_name)
    final_frame = process_subset(llm, nsamples=nsamples)

    # Check if final_frame is not None before creating results directory
    if final_frame is None:
        logger.warning("No final frame to process. Skipping results directory creation.")
        return
    
    if not os.path.exists('results'):
        os.makedirs('results')

    # Write the DataFrame to a CSV file
    filename = f"results/medical_meadow_medqa_result_{model_name}.csv"
    final_frame.to_csv(filename, index=False)
    logger.info(f"Results have been saved to {filename}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Medical Meadow MedQA dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.1", help="Model name to use for processing.")
    args = parser.parse_args()

    process_dataset(model_name=args.model_name, nsamples=args.nsamples)
