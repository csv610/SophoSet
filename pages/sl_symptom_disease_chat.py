import streamlit as st
from datasets import load_dataset
import json
from typing import Tuple, Optional
from langchain_ollama import OllamaLLM
from llm_chat import LLMChat

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

def config_panel(disease_mapping: dict) -> Tuple[Optional[dict], int, int, LLMChat]:
    """Handle the contents of the left panel (sidebar)."""
    st.sidebar.title("Navigation")

    dataset = load_data()
    if dataset is None:
        return None, 0, 0, None

    num_items_per_page = st.sidebar.slider("Items per Page", min_value=1, max_value=10, value=5)
    
    total_items = len(dataset)
    total_pages = -(-total_items // num_items_per_page)  # Ceiling division

    selected_page = st.sidebar.selectbox("Page Number", options=range(1, total_pages + 1))
    model_name  = st.sidebar.selectbox("LLM Model", options=['llama3.1'])
    llm = get_llm(model_name)

    return dataset, num_items_per_page, selected_page, llm

def display_question(index: int, row: dict, llm: LLMChat, disease_mapping: dict):
    """Display a single question with its text and corresponding disease name."""
    st.header(f"Question: {index}")

    text = row['text']
    label = row['label']  # The label from the dataset

    st.write(f"Symptoms: {text}")

    # Map the label to the corresponding disease name using the reversed mapping
    disease_name = disease_mapping.get(str(label), "Unknown Disease")
    st.write(f"Disease: {disease_name}")

    # Input box for the user to ask their own question, pre-filled with the default question text
    default_qs = text + " What could be the disease?"
    user_question = st.text_area(f"Ask your own question for Question {index}", value=default_qs, height=100)

    if st.button(f"Ask Question {index}"):
        with st.spinner("Processing ..."):
            response = llm.get_answer(user_question)
            st.write("Response:", response)

    st.divider()

def view_dataset():
    """Main function to display the dataset viewer."""
    st.title("Dataset: Symptom Disease")

    # Load the disease name mapping from the JSON file
    disease_mapping = load_disease_mapping()

    dataset, num_items_per_page, selected_page, llm = config_panel(disease_mapping)

    if dataset is None:
        return

    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, len(dataset))

    for i in range(start_index, end_index):
        display_question(i + 1, dataset[i], llm, disease_mapping)

    col1, col2 = st.columns(2)
    with col1:
        if selected_page > 1:
            if st.button("Previous Page"):
                st.session_state.page = selected_page - 1
                st.rerun()
    with col2:
        if selected_page < -(-len(dataset) // num_items_per_page):
            if st.button("Next Page"):
                st.session_state.page = selected_page + 1
                st.rerun()

if __name__ == "__main__":
    view_dataset()

