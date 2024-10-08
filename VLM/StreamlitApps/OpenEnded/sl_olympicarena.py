import time 

import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
from vlm_chat import LlavaModel
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # For consistent results

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(subject):
    ds = load_dataset("GAIR/OlympicArena", subject)
    return ds

# Function to load image from a URL
def load_image(image_url):
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))

@st.cache_resource 
def load_vlm_model():
    vlm = LlavaModel()
    return vlm

def build_prompt(question, options=None):
    if not options:  # Check if options are empty
        prompt = f"You are an expert in mathematics. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are an expoert in mathematics. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
    return prompt

def ask_vlm(params): 
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

def config_panel() -> dict:
    st.sidebar.title("OlympicArena")

    config = {
        "dataset": None,
        "start_index": 0,
        "end_index": 0,
        "vlm_model": "llava"
    }

    subject = st.sidebar.selectbox("Select Subject", ["Astronomy", "Biology", "CS", "Chemistry", "Geography", "Math", "Physics"])
    
    dataset = load_data(subject)
    if dataset is None:
        st.error("Dataset could not be loaded. Please try again.")
        return config 

    dataset = dataset['test'] 

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

    # Select the VLM model
    config["vlm_model"] = st.sidebar.selectbox("Select VLM Model", ["llama3.2", "llama3.1"])  # Model selection

    return config

def process_question(row, index):
    st.header(f"Question #{index}")

    # Display instruction
    question = row['problem']
    st.write(question)
    
    # Detect the language of the question
    try:
        language = detect(question)            
        if language != 'en':
            st.write(f"Detected language: {language}")
    except Exception as e:
        st.error(f"Error detecting language: {str(e)}")

    # Display images
    image = None
    if row['figure_urls']:
        for url in row['figure_urls']:
            try:
                image = load_image(url)
                st.image(image, caption=f"Item {index}", use_column_width=True)
            except Exception as e:
                st.write(f"Unable to load image from {url}: {e}")

    vlm_params = {
        "question": question,
        "options": None, 
        "image": image,
        "index": index
    }

    ask_vlm(vlm_params) 
    
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
        process_question(dataset[i], i+1)
      
if __name__ == "__main__":
    process_dataset()

