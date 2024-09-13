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

def process_dataset(model_name, nsamples=None):
    splits = ["train", "validation", "test"]
    
    dataframes = []
    llm = LLMChat(model_name)

    for split in splits:
        print(f"Processing split: {split}")
        df = process_subset(llm,split, nsamples)
        dataframes.append(df)
 
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Check if combined DataFrame is empty
    if combined_df.empty:
        logging.warning("The combined DataFrame is empty. No data to write.")
        return

    # Write the combined DataFrame to a CSV file
    filename = f"sciq_result_{model_name}.csv"  # Create filename with model name
    combined_df.to_csv(filename, index=False)
    logging.info(f"Combined DataFrame written to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SCiQ dataset")
    parser.add_argument("-m", "--model", type=str, default="llama3.1", required=True, help="Model name to use for processing.")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    # Removed logging level argument
    # Set default logging level
    numeric_level = logging.INFO
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Starting process with {args.nsamples if args.nsamples else 'all'} samples")
    process_dataset(model_name=args.model, nsamples=args.nsamples)
    logging.info("Process completed")

