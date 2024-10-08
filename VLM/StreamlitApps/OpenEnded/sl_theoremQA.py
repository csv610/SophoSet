import time 

import streamlit as st
from datasets import load_dataset

from vlm_chat import LlavaModel

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data():
    try:
        ds = load_dataset("TIGER-Lab/TheoremQA")
        return ds['test']
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None  # Return None or handle as needed

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

def ask_vlm(params):  # Accept a single dictionary parameter
    question = params["question"]
    options = params["options"]
    image = params["image"]
    index = params["index"]

    # Use a button to trigger the answer retrieval
    if st.button(f"Ask VLM #{index}"):
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

def config_panel():
    config = {
        "dataset": None,
        "start_index": 0,
        "end_index": 0,
        "vlm_model": "llava"
    }

    st.sidebar.title("TheoremQA")
    dataset = load_data()  # Load the dataset
    if dataset is None:
        st.error("Dataset could not be loaded. Please try again.")
        return config  # Return the initial config if dataset is None

    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    total_items = len(dataset)  # Assuming you want the 'test' split
    total_pages = (total_items + num_items_per_page - 1) // num_items_per_page

    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, total_items)

    config["dataset"] = dataset 
    config["start_index"] = start_index
    config["end_index"] = end_index

    # Add model selection in the sidebar
    vlm_model_options = ["llama3.2", "llama3.1"]
    config["vlm_model"] = st.sidebar.selectbox("Select VLM Model", options=vlm_model_options)

    return config  

def process_question(row, index):
    st.header(f"Question #{index}")

    # Display question and answer
    question = row['Question']
    st.write(question)

    image = None
    if row['Picture']:
        image = row['Picture']
        st.image(image)  # Display the image

    if st.button(f"Show Correct Answer #{index}"):
        answer = row['Answer']
        st.write(answer)

    vlm_params = {
        "question": question,
        "options": None, 
        "image": image,
        "index": index
    }

    ask_vlm(vlm_params)  

    st.divider()  # Divider between items

# Streamlit app
def process_dataset():   
    config = config_panel()  # Get the config as a dictionary

    dataset = config["dataset"]
   
    if dataset is None:  # Check if dataset is None
        st.error("Dataset could not be loaded. Please try again.")
        return  # Exit the function if dataset is not loaded
    
    start_index = config["start_index"]
    end_index = config["end_index"]
    
    for i in range(start_index, end_index):
        process_question(dataset[i], i+1)

if __name__ == "__main__":
    process_dataset()

