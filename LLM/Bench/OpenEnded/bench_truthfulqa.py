import random
import argparse
import pandas as pd
import logging

from datasets import load_dataset
from llm_chat import LLMChat
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(subset, split):
    ds = load_dataset("truthfulqa/truthful_qa", subset)
    return ds[split]

def process_subset(llm, subset, split, nsamples=None):
    dataset = load_data(subset, split)
    results = []

    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))

    for i in tqdm(indices, desc=f"Processing {subset}", unit="question"):
        row = dataset[i]
        question = row['question']
        
        try:
            llm_response = llm.get_answer(question)
            llm_answer = llm_response['answer']
        except Exception as e:
            logger.error(f"Error generating response for question: {question}")
            logger.error(f"Error details: {str(e)}")
            llm_answer = "Error"

        result = {
            "Question": question,
            "llama3.1": llm_answer
        }

        if subset == "generation":
            result.update({
                "Best Answer": row['best_answer'],
                "Correct Answers": ", ".join(row['correct_answers']),
                "Incorrect Answers": ", ".join(row['incorrect_answers'])
            })
        elif subset == "multiple_choice":
            result.update({
                "MC1 Choices": ", ".join(row['mc1_targets']['choices']),
                "MC1 Labels": ", ".join(map(str, row['mc1_targets']['labels'])),
                "MC2 Choices": ", ".join(row['mc2_targets']['choices']),
                "MC2 Labels": ", ".join(map(str, row['mc2_targets']['labels']))
            })

        results.append(result)

    return pd.DataFrame(results)

def process_dataset(nsamples=None):
    logger.info("Starting TruthfulQA dataset processing")

    subsets = ['generation', 'multiple_choice']
    split = 'validation'

    llm = LLMChat("llama3.1")
    
    frames = []
    for subset in subsets:
        logger.info(f"Processing subset: {subset}")
        df = process_subset(llm, subset, split, nsamples)
        frames.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(frames, ignore_index=True)

    # Write to CSV
    output_file = 'truthfulqa_result.csv'
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Results written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TruthfulQA dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    args = parser.parse_args()

    logger.info(f"Starting script with nsamples={args.nsamples}")
    process_dataset(nsamples=args.nsamples)
    logger.info("Script completed successfully")

