import pandas as pd
import argparse
import random

from datasets import load_dataset
from llm_chat import LLMChat
from tqdm import tqdm

def process_subset(llm, split, nsamples = None):
    # Load the dataset inside the function
    dataset = load_dataset("automated-research-group/winogrande")[split]

    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))
    
    # Create a list to store the data
    data = []
    
    for i in tqdm(indices, desc="Processing items"):
        row = dataset[i]
        question = row['request']

        try:
            llm_response = llm.get_answer(question)
            llm_answer = llm_response['answer']
        except Exception as e:
            print(f"Error generating response for question: {question}")
            print(f"Error details: {str(e)}")
            llm_answer = "Error"

        data.append({
            "Question": i,
            "Request": question,
            "Response": row['response'],
            "llama3.1": llm_answer
        })
    
    # Create a pandas DataFrame
    df = pd.DataFrame(data)
    return df

def process_dataset(nsamples=None):
    llm = LLMChat("llama3.1")
    df= process_subset(llm, "validation", nsamples)

    # Save the DataFrame to a CSV file
    df.to_csv("winogrande_result.csv", index=False)
    print("DataFrame saved to winogrande_result.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Winogrande dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    args = parser.parse_args()

    process_dataset(nsamples=args.nsamples)
