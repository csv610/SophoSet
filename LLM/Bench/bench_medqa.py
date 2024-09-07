from datasets import load_dataset
import pandas as pd
from llm_chat import LLMChat
from tqdm import tqdm

def load_data(split):
    model_name = "openlifescienceai/medqa"
    ds = load_dataset(model_name)
    return ds[split]

def process_subset(llm, split):
    dataset = load_data(split)
    results = []

    nsamples = len(dataset)

    for i in tqdm(range(nsamples), desc=f"Processing {split} dataset"):
        row  = dataset[i]
        data = row['data']
        question = data.get('Question', 'No question available')
        options  = data.get('Options', {})
        answer   = data.get('Correct Option', 'N/A')
        
        try:
            llm_response = llm.get_answer(question, options)
            llm_answer = llm_response['answer']
        except Exception as e:
            print(f"Error processing question {i+1}: {str(e)}")
            llm_answer = "Error occurred"
        
        result = {
            'id': f"{split}_{i + 1}",
            'Question': question,
            'Options': ', '.join([f"{k}: {v}" for k, v in options.items()]),
            'Correct Answer': answer, 
            'llama3.1': llm_answer
        }
        results.append(result)

    return pd.DataFrame(results)

def process_dataset():
    splits = ["train", "test", "dev"]
    frames = []
    llm = LLMChat("llama3.1")

    for split in splits:
        df = process_subset(llm, split)
        df['Split'] = split  # Add a column to identify the split
        frames.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(frames, ignore_index=True)

    # Save combined DataFrame to CSV
    output_file = "medqa_results.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved all results to {output_file}")

if __name__ == "__main__":
    process_dataset()

