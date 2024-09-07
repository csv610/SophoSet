import random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from llm_chat import LLMChat

def load_data(split):
    ds = load_dataset("allenai/sciq")
    return ds[split]

def process_subset(llm, split):
    dataset = load_data(split)
    data = []
    for i in tqdm(range(len(dataset)), desc=f"Processing {split} split"):
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

def process_dataset():
    splits = ["train", "validation", "test"]
    dataframes = []
    llm = LLMChat("llama3.1")

    for split in splits:
        print(f"Processing split: {split}")
        df = process_subset(llm,split)
        dataframes.append(df)
 
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Write the combined DataFrame to a CSV file
    combined_df.to_csv("sciq_result.csv", index=False)
    print("Combined DataFrame written to sciq_result.csv")

if __name__ == "__main__":
    process_dataset()

