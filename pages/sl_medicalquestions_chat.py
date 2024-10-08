import streamlit as st
from datasets import load_dataset
import json
from typing import Tuple, Optional
from langchain_ollama import OllamaLLM
from llm_chat import LLMChat

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

@st.cache_resource
def get_llm(model_name: str):
    """Initialize and cache the LLM."""
    return LLMChat(model_name)

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
    llm = get_llm(model_name)

    return dataset, num_items_per_page, selected_page, llm

def display_question(index: int, row: dict, llm: LLMChat):
    """Display a single question with its text and corresponding disease name."""
    st.header(f"Question: {index}")

    question = row['text']
    st.write(question)

    if st.button(f"Answer Question {index}"):
        with st.spinner("Processing ..."):
            response = llm.get_answer(question)
            st.write("Response:", response)

    user_question = st.text_area(f"Ask your own question for Question {index}", height=100)

    if st.button(f"Ask Question {index}"):
        with st.spinner("Processing ..."):
            prompt = "Based on the Context of " + question + " Answer the User Question: " + user_question
            response = llm.get_answer(prompt)
            st.write("Response:", response)

    st.markdown("---")

def view_dataset():
    """Main function to display the dataset viewer."""
    st.title("Medical Questions")

    dataset, num_items_per_page, selected_page, llm = config_panel()

    if dataset is None:
        return

    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, len(dataset))

    for i in range(start_index, end_index):
        display_question(i + 1, dataset[i], llm)

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

