import streamlit as st
from datasets import load_dataset
import json
from typing import Tuple, Optional
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
def load_llm_model(model_name: str):
    """Initialize and cache the LLM."""
    return LLMChat(model_name)

def talk_to_llm(i, row, model_name):
    question = row['text']

    llm = load_llm_model(model_name)  # Changed from upload_llm_model to load_llm_model

    if st.button(f"LLM Answer:{i+1}"):
        with st.spinner("Processing ..."):

            response = llm.get_answer(question)
            st.write(f"Answer: {response['answer']}")
            st.write(f"Number of Input words: {response['num_input_words']}")
            st.write(f"Number of output  words: {response['num_output_words']}")
            st.write(f"Time: {response['response_time']}")

    if st.button(f"Ask Question: {i+1}"):
        user_question = st.text_area(f"Edit Question {i+1}", height=100)
        if user_question is not None:
              print( "USER QUESTION; ", user_question)
              with st.spinner("Processing ..."):
                   prompt = "Based on the Context of " + question + " Answer the User Question: " + user_question
                   response = llm.get_answer(prompt)
                   st.write(f"Answer: {response['answer']}")
                   st.write(f"Number of Input words: {response['num_input_words']}")
                   st.write(f"Number of output  words: {response['num_output_words']}")
                   st.write(f"Time: {response['response_time']}")


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
    for i in range(start_index, end_index):
        row = dataset[i]

        st.header(f"Question: {i+1}")

        question = row['text']
        st.write(question)

        talk_to_llm(i, row, model_name)
        st.markdown("---")


if __name__ == "__main__":
    view_dataset()

