import streamlit as st
from datasets import load_dataset
import json
from typing import Tuple, Optional
from llm_chat import load_llm_model, ask_llm

st.set_page_config(layout="wide")

@st.cache_data
def load_data(split: str = "train"):
    """Load and cache the dataset."""
    try:
        ds = load_dataset("fhirfly/medicalquestions")
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def config_panel() -> Tuple[Optional[dict], int, int, LLMChat]:
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
    
    return dataset, num_items_per_page, selected_page

def view_dataset():
    """Main function to display the dataset viewer."""
    st.title("Dataset : Medical Questions")

    dataset, num_items_per_page, selected_page = config_panel()

    if dataset is None:
        return 

    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, len(dataset))

    model_name = "llama3.1"
    llm = load_llm_model(model_name)
    
    for i in range(start_index, end_index):
        row = dataset[i]
        question = row['text']

        st.header(f"Question: {i+1}")
        st.write(question)

        ask_llm(llm, question, i+1)

        st.divider()

if __name__ == "__main__":
    view_dataset()
