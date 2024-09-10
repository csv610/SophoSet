import logging
import random
import argparse
import pandas as pd

from tqdm import tqdm
from llm_chat import LLMChat
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_subset(llm, split="train", nsamples=None):
    dataset = load_dataset("openlifescienceai/medmcqa")[split]
    
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))

    model_name = llm.get_model_name()
    
    data = []
    for i in tqdm(indices, desc="Processing questions"):
        row = dataset[i]
        question = row['question']
        options = [row['opa'], row['opb'], row['opc'], row['opd']]
        answer = row['cop']
        explanation = row['exp']

        try:
            llm_response = llm.get_answer(question, options)
            llm_answer = llm_response['answer']
        except Exception as e:
            logging.error(f"Error processing question {i}: {str(e)}")
            llm_answer = "Error"

        data.append({
            "id": f"{split}_{i+1}",
#            "question": question,
#            "options": options,
            "answer": answer,
#            "explanation": explanation,
            model_name: llm_answer
        })
    
    return pd.DataFrame(data)

def process_dataset(nsamples=None):
    logging.info("Dataset: Medmcqa")

    model_name = "llama3.1"
    llm = LLMChat(model_name)

    df = process_subset(llm, nsamples=nsamples)
    if df is not None:
        csv_filename = "medmcqa_result.csv"
        df.to_csv(csv_filename, index=False)
        logging.info(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Medmcqa dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    args = parser.parse_args()

    process_dataset(nsamples=args.nsamples)

