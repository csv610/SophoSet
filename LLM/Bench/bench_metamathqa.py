import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from llm_chat import LLMChat

def load_data(split='train'):
    try:
        ds = load_dataset("meta-math/MetaMathQA")
        if split not in ds:
            raise ValueError(f"Invalid split: {split}. Available splits are: {', '.join(ds.keys())}")
        return ds[split]
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None


def process_subset(llm, split='train'):
    dataset = load_data(split)
    if dataset is None:
        return None

    subset_data = []
    nsamples = len(dataset)
    for i in tqdm(range(nsamples), desc="Processing ..."):
        row = dataset[i]
        llm_response = llm.get_answer(row['query'], row['response'])
        llm_answer = llm_response['answer']
        subset_data.append({
            'id': f"{split}_{i + 1}",
            'Query': row['query'],
            'Response': row['response'],
            'llama3.1': llm_answer
        })
    return pd.DataFrame(subset_data)

def process_dataset():
    print("Dataset: MetaMathQA")
    llm = LLMChat("llama3.1")

    # Process the dataset
    df = process_subset(llm)
    if df is None:
        return

    # Write DataFrame to CSV
    csv_filename = "metamathqa_result.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Results written to {csv_filename}")

if __name__ == "__main__":
    process_dataset()
