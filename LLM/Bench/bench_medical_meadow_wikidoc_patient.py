from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from llm_chat import LLMChat

def load_data(split):
    try:
        ds = load_dataset("medalpaca/medical_meadow_wikidoc_patient_information")
        return ds[split]
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def process_subset(llm, split="train"):
    dataset = load_data(split)
    
    if dataset is None:
        print("Unable to load the dataset. Please try again later.")
        return None

    nsamples = len(dataset)
    data = []

    for i in tqdm(range(nsamples), desc="Processing dataset"):
        row = dataset[i]
        question = row['input']

        try:
            llm_response = llm.get_answer(question)
            llm_answer = llm_response['answer']
        except Exception as e:
            print(f"Error getting LLM answer for question {i + 1}: {str(e)}")
            llm_answer = "Error: Unable to get answer"

        data.append({
            'id': f"{split}_{i + 1}",
            'Question': question,
            'Answer': row['output'],
            'llama3.1': llm_answer
        })

    df = pd.DataFrame(data)
    return df

def process_dataset():
    print("Dataset: Medical Meadow Wikidoc Patient")
    llm = LLMChat("llama3.1")
    df = process_subset(llm)
    
    if df is not None:
        # Save the DataFrame to a CSV file
        df.to_csv('medical_meadow_wikidoc_patient_result.csv', index=False)
        print("DataFrame saved to medical_meadow_wikidoc_patient_result.csv")
    else:
        print("Failed to process the dataset.")

if __name__ == "__main__":
    process_dataset()
