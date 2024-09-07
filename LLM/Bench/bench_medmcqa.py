from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from llm_chat import LLMChat

def process_subset(llm, split="train"):
    dataset = load_dataset("openlifescienceai/medmcqa")[split]
    
    data = []
    for i in tqdm(range(len(dataset)), desc="Processing questions"):
        row = dataset[i]
        question = row['question']
        options = [row['opa'], row['opb'], row['opc'], row['opd']]
        correct_answer = row['cop']
        explanation = row['exp']

        try:
            llm_response = llm.get_answer(question, options)
            llm_answer = llm_response['answer']
        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
            llm_answer = "Error"

        data.append({
            "id": f"{split}_{i + 1}",
            "question": question,
            "options": options,
            "correct_answer": answer,
            "explanation": explanation,
            "llama3.1": llm_answer
        })
    
    return pd.DataFrame(data)

def process_dataset():
    print("Dataset: Medmcqa dataset")
    llm = LLMChat("llama3.1")
    df = process_subset(llm)
    if df is not None:
        csv_filename = "medmcqa_result.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    process_dataset()

