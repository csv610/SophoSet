import json
import pandas as pd
from datasets import load_dataset
from langchain_ollama import OllamaLLM
from llm_chat import LLMChat
from tqdm import tqdm

def load_data(split: str = "train"):
    """Load the dataset."""
    try:
        ds = load_dataset("fhirfly/medicalquestions")
        return ds[split]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def process_subset(llm, split='train'):

    dataset = load_data(split)
    if dataset is None:
        return

    data = []
    for i, row in tqdm(enumerate(dataset), total=len(dataset), desc="Processing questions"):
        question = row['text']
        response = llm.get_answer(question)

        user_question = "What is the treatment?"  # Example user question
        prompt = "Based on the Context of " + question + " Answer the User Question: " + user_question
        try:
            llm_response = llm.get_answer(prompt)
            llm_answer = llm_response['answer']
        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
            llm_answer = "Error: Unable to process this question"

        data.append({
            "Question": question,
            "llama3.1": llm_answer
        })

        df = pd.DataFrame(data)
        return df

def process_dataset():
    print("Dataset: Medical Questions")
    llm = LLMChat("llama3.1")
    df = process_subset(llm)
    if df is not None:
        df.to_csv("medicalquestions_result.csv", index=False)
        print("Data saved to medicalquestions_result.csv")

if __name__ == "__main__":
    process_dataset()