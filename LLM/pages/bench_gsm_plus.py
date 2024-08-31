import pandas as pd
from datasets import load_dataset
from llm_chat import LLMChat
from tqdm import tqdm

# Define SPLITS constant
SPLITS = ["test", "testmini"]

def load_data(split):
    model_name = "qintongli/GSM-Plus"
    try:
        ds = load_dataset(model_name)
        if split not in ds:
            raise ValueError(f"Split '{split}' not found in the dataset.")
        return ds[split]
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def process_dataset():
    all_data = []
    llm = LLMChat("llama3.1")

    for split in SPLITS:
        dataset = load_data(split)
        if dataset is None:
            continue

        nsamples = len(dataset)
        for i in tqdm(range(nsamples), desc=f"Processing {split}"):
            row = dataset[i]
            question = row['question']
            answer = row['answer']
            
            try:
                llm_response = llm.get_answer(question)
                llm_answer = llm_response['answer']
            except Exception as e:
                print(f"Error getting LLM response for question {i+1} in {split}: {str(e)}")
                llm_answer = "Error: Failed to get LLM response"

            all_data.append({
                'split': split,
                'question_id': i + 1,
                'question': question,
                'correct_answer': answer,
                'llm_answer': llm_answer
            })

    return pd.DataFrame(all_data)

def main():
    print("Starting dataset processing...")
    df = process_dataset()
    df.to_csv('gsm_plus_results.csv', index=False)
    print("Results saved to 'gsm_plus_results.csv'")

if __name__ == "__main__":
    main()
