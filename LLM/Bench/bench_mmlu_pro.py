import string
import pandas as pd
from datasets import load_dataset
from llm_chat import LLMChat

def process_subset(llm, split):
    
    try:
        subset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    except Exception as e:
        print(f"Error loading dataset for split '{split}': {str(e)}")
        return None
    
    data = []
    for i, row in enumerate(subset):
        question = row['question']
        choices = row['options']
        answer = row['answer']

        try:
            llm_response = llm.get_answer(question, choices)
            llm_answer = llm_response['answer']
        except Exception as e:
            print(f"Error getting answer for question {i}: {str(e)}")
            llm_answer = "Error"

        data.append({
            'Question': question,
            'Choices': choices,
            'Answer': answer,
            'llama3.1': llm_answer
        })
    
    df = pd.DataFrame(data)
    return df

def process_dataset():
    print("Dataset: MMLU Pro")
    
    llm = LLMChat("llama3.1")
    
    # List of available splits
    splits = ["test", "validation"]
    
    # Collect all DataFrames
    dataframes = []
    for split in splits:
        df = process_subset(llm, split)
        if df is not None:
            dataframes.append(df)
    
    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Write the combined DataFrame to a CSV file
    combined_df.to_csv("mmlu_pro_result.csv", index=False)
    print("Combined DataFrame written to mmlu_pro_result.csv")

if __name__ == "__main__":
    process_dataset()
