import random
import argparse
import pandas as pd
import logging

from tqdm import tqdm
from llm_chat import LLMChat
from datasets import load_dataset

def load_data(split):
    ds = load_dataset("allenai/sciq")
    return ds[split]

def process_subset(llm, split, nsamples=None):
    dataset = load_data(split)
    data = []

    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))

    for i in tqdm(indices, desc=f"Processing {split} split"):
        row = dataset[i]
        options = [row['correct_answer'], row['distractor1'], row['distractor2'], row['distractor3']]
        random.shuffle(options)

        try:
            llm_response = llm.get_answer(row['question'], options)
            llm_answer = llm_response['answer']
        except Exception as e:
            print(f"Error getting answer for question {i}: {e}")
            llm_answer = "Error"

        data.append({
            "id": f"{split}_{i + 1}",
            "Question": row['question'],
            "Options": options,
            "Answer": row['correct_answer'],
            "Explanation": row['support'],
            "llama3.1": llm_answer
        })
    
    return pd.DataFrame(data)

def process_dataset(nsamples=None):
    splits = ["train", "validation", "test"]
    dataframes = []
    llm = LLMChat("llama3.1")

    for split in splits:
        print(f"Processing split: {split}")
        df = process_subset(llm,split, nsamples)
        dataframes.append(df)
 
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Write the combined DataFrame to a CSV file
    combined_df.to_csv("sciq_result.csv", index=False)
    logging.info("Combined DataFrame written to sciq_result.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SCiQ dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    parser.add_argument("-l", "--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    args = parser.parse_args()

    # Set up logging
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log}")
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Starting process with {args.nsamples if args.nsamples else 'all'} samples")
    process_dataset(nsamples=args.nsamples)
    logging.info("Process completed")

