import pandas as pd
from datasets import load_dataset
from llm_chat import LLMChat

def load_data(subset, split):
    ds = load_dataset("truthfulqa/truthful_qa", subset)
    return ds[split]

def process_subset(llm, subset, split):
    dataset = load_data(subset, split)
    results = []

    for i, row in enumerate(dataset):
        question = row['question']
        
        try:
            llm_response = llm.generate(question)
            llm_answer = llm_response['answer']
        except Exception as e:
            print(f"Error generating response for question: {question}")
            print(f"Error details: {str(e)}")
            llm_answer = "Error"

        result = {
            "Question": question,
            "llama3.1": llm_answer
        }

        if subset == "generation":
            result.update({
                "Best Answer": row['best_answer'],
                "Correct Answers": ", ".join(row['correct_answers']),
                "Incorrect Answers": ", ".join(row['incorrect_answers'])
            })
        elif subset == "multiple_choice":
            result.update({
                "MC1 Choices": ", ".join(row['mc1_targets']['choices']),
                "MC1 Labels": ", ".join(map(str, row['mc1_targets']['labels'])),
                "MC2 Choices": ", ".join(row['mc2_targets']['choices']),
                "MC2 Labels": ", ".join(map(str, row['mc2_targets']['labels']))
            })

        results.append(result)

    return pd.DataFrame(results)

def process_dataset():
    print("Dataset: TruthfulQA")

    subsets = ['generation', 'multiple_choice']
    split = 'validation'

    llm = LLMChat("llama3.1")
    
    frames = []
    for subset in subsets:
        df = process_subset(llm, subset, split)
        frames.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(frames, ignore_index=True)

    # Write to CSV
    combined_df.to_csv('truthfulqa_result.csv', index=False)
    print("Results written to truthfulqa_result.csv")

if __name__ == "__main__":
    process_dataset()



