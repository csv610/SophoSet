import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from llm_chat import LLMChat

MODEL_ID = "lighteval/MATH"
SUBSETS = ["all", "algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
SPLITS = ["train", "test"]

def load_data(subset, split):
    try:
        ds = load_dataset(MODEL_ID, subset)
        return ds[split]
    except Exception as e:
        print(f"Error loading dataset {subset} - {split}: {str(e)}")
        return None

def process_datasets():
    llm = LLMChat("llama3.1")
    all_data = []

    for subset in SUBSETS:
        for split in SPLITS:
            dataset = load_data(subset, split)
            if dataset is None:
                continue

            for i in tqdm(range(len(dataset)), desc=f"Processing {subset} - {split}"):
                row = dataset[i]
                problem = row['problem']
                try:
                    llm_response = llm.get_answer(problem)
                    llm_solution = llm_response['explanation']
                except Exception as e:
                    print(f"Error getting LLM response for {subset} - {split}, index {i}: {str(e)}")
                    llm_solution = "Error"

                all_data.append({
                    'subset': subset,
                    'split': split,
                    'index': i,
                    'problem': problem,
                    'solution': row['solution'],
                    'llama3.1': llm_solution
                })

    df = pd.DataFrame(all_data)
    return df

if __name__ == "__main__":
    result_df = process_datasets()
    print(result_df.info())
    
    # Save the DataFrame to a CSV file
    result_df.to_csv('math_dataset.csv', index=False)
    print("Data saved to math_dataset.csv")
