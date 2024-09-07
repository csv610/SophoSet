import streamlit as st
from datasets import load_dataset
import json
from typing import Tuple, Optional
from langchain_ollama import OllamaLLM
from llm_chat import LLMChat
import pandas as pd

st.set_page_config(layout="wide")

# Load the JSON file containing the disease-to-label mapping and reverse it
@st.cache_data
def load_disease_mapping(json_path: str = 'disease_labels.json') -> dict:
    """Load and cache the reversed disease name mapping."""
    try:
        with open(json_path, 'r') as file:
            disease_to_label = json.load(file)
        # Reverse the dictionary to get label-to-disease mapping
        label_to_disease = {str(v): k for k, v in disease_to_label.items()}
        return label_to_disease
    except Exception as e:
        st.error(f"Error loading disease mapping: {e}")
        return {}

@st.cache_data
def load_data(split: str = "train"):
    """Load and cache the dataset."""
    try:
        ds = load_dataset("duxprajapati/symptom-disease-dataset")
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_resource
def get_llm(model_name: str):
    """Initialize and cache the LLM."""
    return LLMChat(model_name)

def process_subset(llm):
    """Main function to display the dataset viewer."""
    print("Dataset: Symptom Disease")

    # Load the disease name mapping from the JSON file
    disease_mapping = load_disease_mapping()

    # Load the dataset
    dataset = load_data()
    if dataset is None:
        return

    # Create a list to store the results
    results = []

    for i, row in enumerate(dataset):
        index = i + 1
        text = row['text']
        label = row['label']  # The label from the dataset

        # Map the label to the corresponding disease name using the reversed mapping
        disease_name = disease_mapping.get(str(label), "Unknown Disease")
        try:
            llm_response = llm.get_answer(text)
            llm_answer = llm_response['answer']
        except Exception as e:
            print(f"Error getting answer for question {i}: {e}")
            llm_answer = "Error"

        # Append the result to the list
        results.append({
            "Question": index,
            "Symptoms": text,
            "Disease": disease_name,
            "llama3.1": llm_answer
        })

    return pd.DataFrame(results)

def process_dataset():
    llm = LLMChat("llama3.1")
    df = process_subset(llm)
    df.to_csv('symptom_disease_result.csv', index=False)

if __name__ == "__main__":
    process_subset()

