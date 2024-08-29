from datasets import load_dataset
import pandas as pd
from llm_chat import LLMChat
from tqdm import tqdm

def load_data(split):
    model_name = "openlifescienceai/medqa"
    ds = load_dataset(model_name)
    return ds[split]

def process_dataset(llm, split):
    dataset = load_data(split)
    results = []

    nsamples = len(dataset)
    nsamples = min(100, nsamples)
   
    for i in tqdm(range(nsamples), desc=f"Processing {split} dataset"):
        row  = dataset[i]
        data = row['data']
        question = data.get('Question', 'No question available')
        options  = data.get('Options', {})
        answer   = data.get('Correct Option', 'N/A')
        llm_answer = llm.get_answer(question, options)
        
        result = {
            'Question Number': i + 1,
            'Question': question,
            'Options': ', '.join([f"{k}: {v}" for k, v in options.items()]),
            'Correct Answer': answer, 
            'LLM Answer': llm_answer
        }
        results.append(result)

    return pd.DataFrame(results)

def main():
    splits = ["train", "test", "dev"]
    all_results = []
    llm = LLMChat("llama3.1")

    for split in splits:
        df = process_dataset(llm, split)
        df['Split'] = split  # Add a column to identify the split
        all_results.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save combined DataFrame to CSV
    output_file = "medqa_results.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved all results to {output_file}")

if __name__ == "__main__":
    main()

