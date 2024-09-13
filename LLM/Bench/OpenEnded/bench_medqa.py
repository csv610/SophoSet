import logging
import argparse
import random
import pandas as pd

from tqdm import tqdm
from llm_chat import LLMChat
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(split):
    model_name = "openlifescienceai/medqa"
    try:
        ds = load_dataset(model_name)
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None
    return ds[split]

def process_subset(llm, split, nsamples=None):
    dataset = load_data(split)
    results = []
    
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))

    logger.info(f"Processing {split} dataset with {len(indices)} samples")

    for i in tqdm(indices, desc=f"Processing {split} dataset"):
        row  = dataset[i]
        data = row['data']
        question = data.get('Question', 'No question available')
        options  = data.get('Options', {})
        answer   = data.get('Correct Option', 'N/A')
        
        try:
            llm_response = llm.get_answer(question, options)
            llm_answer = llm_response['answer']
        except Exception as e:
            logger.error(f"Error processing question {i+1}: {str(e)}")
            llm_answer = "Error occurred"
        
        result = {
            'id': f"{split}_{i + 1}",
            'Question': question,
            'Options': ', '.join([f"{k}: {v}" for k, v in options.items()]),
            'Correct Answer': answer, 
            'llama3.1': llm_answer
        }
        results.append(result)

    logger.info(f"Finished processing {split} dataset")
    return pd.DataFrame(results)

def process_dataset(model_name, nsamples=None):
    splits = ["train", "test", "dev"]
    frames = []
    llm = LLMChat(model_name)

    logger.info(f"Starting dataset processing with {nsamples if nsamples else 'all'} samples per split")

    for split in splits:
        df = process_subset(llm, split, nsamples=nsamples)
        df['Split'] = split  # Add a column to identify the split
        frames.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(frames, ignore_index=True)

    # Save combined DataFrame to CSV
    output_file = f"medqa_result_{model_name}.csv"
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Saved all results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Medqa dataset")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.1", required=True, help="Name of the model to use.")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    args = parser.parse_args()

    logger.info(f"Starting script with nsamples={args.nsamples}")
    process_dataset(model_name=args.model_name, nsamples=args.nsamples)
    logger.info("Script execution completed")

