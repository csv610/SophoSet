import time 

import streamlit as st
from datasets import load_dataset

from vlm_chat import LlavaModel
from io import BytesIO

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    ds = load_dataset("lmms-lab/OlympiadBench")
    ds = ds[split]
    return ds

@st.cache_resource
def load_vlm_model():
    vlm = LlavaModel()
    return vlm

def build_prompt(question, options=None):
    if not options:  # Check if options are empty
        prompt = f"You are an expert in mathematics. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are an expert in mathematics. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
    return prompt

def ask_vlm(question, options, image, index):
    """Retrieve an answer from the VLM model."""
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
                st.error(f"Error retrieving answer: {str(e)}")
            finally:
                st.session_state.processing = False  # Reset processing state

def process_question(row, index):
    st.header(f"Question #{index}")

    # Display instruction
    question = row['question']
    st.write(question)

    images = row.get('images', [])
    for image in images:
        try:
            st.image(image, caption=f"Item {index}", width=500)
        except Exception as e:
            st.write(f"Unable to load image: {e}")

    ask_vlm(question, None, images, index)
    st.divider()

# Streamlit app
def config_panel():
    st.sidebar.title("OlympiadBench")

    config = {
        "dataset": None,
        "start_index": 0,
        "end_index": 0
    }

    split = st.sidebar.selectbox("Select Split", ["test_en", "test_cn"])

    dataset = load_data(split)
    config["dataset"] = dataset  # Update the config with the loaded dataset

    if dataset:
        num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
        total_items = len(dataset)
        total_pages = (total_items + num_items_per_page - 1) // num_items_per_page  # Improved page calculation

        page_options = list(range(1, total_pages + 1))
        selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

        start_index = (selected_page - 1) * num_items_per_page
        end_index = min(start_index + num_items_per_page, total_items)
        config["start_index"] = start_index  # Update start_index in config
        config["end_index"] = end_index      # Update end_index in config

    return config  # Return the updated config

def process_dataset():
    config = config_panel()
    
    dataset = config['dataset']
    
    if dataset is None: 
        st.error("Dataset could not be loaded. Please try again.")
        return 
    
    start_index = config['start_index']
    end_index = config['end_index']

    for i in range(start_index, end_index):
        process_question(dataset[i], i + 1)

if __name__ == "__main__":
    process_dataset()

