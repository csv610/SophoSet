import re
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
            question_text = question[:match.start()] + "?"
            choices_dict_str = match.group(1)
            choices_dict = ast.literal_eval(choices_dict_str)
            choices = list(choices_dict.values())
        else:
            question_text = question
            choices = []

        answer = answer.strip()[0]

        try:
            llm_response = llm.get_answer(question_text, choices)
            llm_answer = llm_response['answer']
            logger.debug(f"LLM answer obtained for question {i + 1}")
        except Exception as e:
            logger.error(f"Error getting LLM answer for question {i + 1}: {str(e)}")
            llm_answer = "Error: Unable to get answer"

        data.append({
            'id': f"{split}_{i+1}",
            'Question': question_text,
            'Choices': choices,
            'Answer': answer,
            model_name: llm_answer
        })

    df = pd.DataFrame(data)
    logger.info(f"Processed {len(df)} questions for {split} split")
    return df

def process_dataset(nsamples=None):
    logger.info("Starting dataset processing: medical meadow medqa")

    model_name = "llama3.1"

    llm = LLMChat(model_name)
    final_frame = process_subset(llm, nsamples=nsamples)

    if final_frame is not None:
        # Write the DataFrame to a CSV file
        final_frame.to_csv("medical_meadow_medqa_result.csv", index=False)
        logger.info("Results have been saved to medical_meadow_medqa_result.csv")
    else:
        logger.warning("No results to save. Check for errors in processing.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Medical Meadow MedQA dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    args = parser.parse_args()

    process_dataset(nsamples=args.nsamples)
