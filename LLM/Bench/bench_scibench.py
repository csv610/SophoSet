from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from llm_chat import LLMChat

def load_data(split='train'):
    try:
        ds = load_dataset("xw27/scibench")
        return ds[split]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def process_dataset():
    dataset = load_data()
    if dataset is None:
        return

    data = []
    llm = LLMChat("llama3.1")

    nsamples = len(dataset)
    for i in tqdm(range(nsamples), desc="Processing questions"):
        row = dataset[i]
        problem = row['problem_text']
        
        try:
            llm_response = llm.get_answer(problem)
            llm_answer = llm_response['answer']
        except Exception as e:
            print(f"Error getting answer for question {i + 1}: {e}")
            llm_answer = "Error"
        
        data.append({
            'id': i + 1,
            'problem Text': problem, 
            'answer': row['answer_number'],
            'llama3.1': llm_answer
        })

    df = pd.DataFrame(data)
    df.to_csv('scibench_dataset.csv', index=False)
    print("Dataset saved to scibench_dataset.csv")

if __name__ == "__main__":
    process_dataset()
