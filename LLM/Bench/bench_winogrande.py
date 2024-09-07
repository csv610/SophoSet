import pandas as pd
from datasets import load_dataset
from llm_chat import LLMChat

from tqdm import tqdm

def process_subset(llm, split):
    # Load the dataset inside the function
    dataset = load_dataset("automated-research-group/winogrande")[split]
    nsamples = len(dataset)
    
    # Create a list to store the data
    data = []
    
    for i in tqdm(range(nsamples), desc="Processing items"):
        row = dataset[i]
        question = row['request']

        try:
            llm_response = llm.generate(question)
            llm_answer = llm_response['answer']
        except Exception as e:
            print(f"Error generating response for question: {question}")
            print(f"Error details: {str(e)}")
            llm_answer = "Error"

        data.append({
            "Question": i,
            "Request": question,
            "Response": row['response'],
            "llama3.1": llm_answer
        })
    
    # Create a pandas DataFrame
    df = pd.DataFrame(data)
    return df

def process_dataset():
    llm = LLMChat("llama3.1")
    df= process_subset(llm, "validation")

    # Save the DataFrame to a CSV file
    df.to_csv("winogrande_result.csv", index=False)
    print("DataFrame saved to winogrande_result.csv")

if __name__ == "__main__":
    process_dataset()
