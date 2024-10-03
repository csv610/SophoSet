import re
import time 
from typing import Tuple, List, Optional

import streamlit as st
from datasets import load_dataset
from vlm_chat import LlavaModel

st.set_page_config(layout="wide")

@st.cache_data()
def load_data(split: str = "test"):
    """Load and cache the dataset."""
    try:
        ds = load_dataset("visheratin/realworldqa")
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None
    
@st.cache_resource
def load_vlm_model():
    vlm = LlavaModel()
    return vlm

def build_prompt(question, options):
    if not options:  # Check if options are empty
        prompt = f"You are an intelligent assistant. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are an intelligent assistant. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
    return prompt

def ask_vlm(question, options, image, index):
      # Load model once
    if st.button(f"Ask VLM : {index}"):
        vlm = load_vlm_model()
        prompt = build_prompt(question, options)

        st.session_state.processing = True  # Set processing state
        with st.spinner("Retrieving answer..."):
            start_time = time.time()  # Start the timer
            try:
                answer = vlm.get_answer(prompt, image)  # Pass the byte stream
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                st.write(f"Model answer: {answer}")
                st.write(f"Elapsed time: {elapsed_time:.3f} seconds")  # Display elapsed time
            except Exception as e:
                st.error(f"Error retrieving answer from VLM: {str(e)}")  # More specific error message
            finally:
                st.session_state.processing = False  # Reset processing state
    

def config_panel() -> dict:

    config = {
        "dataset": None,
        "start_index": 0,
        "end_index": 0
    }

    st.sidebar.title("RealworldQA")

    dataset = load_data() 
    if dataset is None:
        st.error("Dataset could not be loaded. Please try again.")
        return config  # Return the initial config if dataset is None

    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    total_items = len(dataset)
    total_pages = (total_items + num_items_per_page - 1) // num_items_per_page

    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, total_items)

    config["dataset"] = dataset  # Update the config with the loaded dataset
    config["start_index"] = start_index
    config["end_index"] = end_index

    return config

def split_text(text: str) -> Tuple[str, List[str]]:
    """Split the text into question and options."""
    if "Please" in text:
        text = text[:text.rfind("Please")].strip()

    parts = text.split("?", 1)
    if len(parts) == 2:
        question = parts[0] + "?"
        options = re.split(r'(?=\b[A-Z]\.\s)', parts[1].strip())
        options = [opt.strip() for opt in options if opt.strip()]
    else:
        question = text
        options = []

    return question, options

def process_question(row: dict, index):
    """Display a single question with its image and answer."""
    st.header(f"Question #{index}")

    question, options = split_text(row['question'])

    st.write(question)

    image = None
    if row['image']:
        image = row['image']
        st.image(image, caption=f"Qs:{index}")
                    
    if options:
        for option in options:
            st.write(option)

    if st.button(f"Show Correct Answer #{index}"):
        st.write(row['answer'])
    
    ask_vlm(question, options, image, index)

    st.divider()

def process_dataset():
    config = config_panel()  
    dataset = config["dataset"]

    if dataset is None:  # Check if dataset is None
        st.error("Dataset could not be loaded. Please try again.")
        return  # Exit the function if dataset is not loaded
    
    start_index = config["start_index"]
    end_index = config["end_index"]

    for i in range(start_index, end_index):
        process_question(dataset[i], i + 1)

if __name__ == "__main__":
    process_dataset()
