import os
import time
from io import BytesIO

from PIL import Image
import streamlit as st
from datasets import load_dataset
from vlm_chat import LlavaChat

# Load the dataset
@st.cache_data()
def load_data():
    ds = load_dataset("BoKelvin/SLAKE")
    return ds

@st.cache_resource
def load_vlm_model():
    vlm = LlavaChat()
    return vlm

def build_prompt(question, options=None):
    if not options:  # Check if options are empty
        prompt = f"You are an expert in medical diagnosis. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are an expert in medical diagnosis. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
    return prompt

def ask_vlm(question, options, image, index):
    # Use a button to trigger the answer retrieval
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

def process_question(row, img_dir, index):
        st.header(f"Question #{index}")

        # Display instruction
        question = row['question']
        st.write(question)

        image = None
        if row['img_name']:
            if not os.path.exists(img_dir):
                st.error(f"Image directory does not exist. Download the images from Huggingface and place in the local folder {img_dir}.")
                return
            image = os.path.join(img_dir, row['img_name'])  # Combine img_dir with the image name
            st.image(image, caption=f"Qs: {index}", use_column_width=True)

        if st.button(f"Show Correct Answer #{index}"):
            st.write(row['answer'])

        ask_vlm(question, None, image, index)

        st.divider()
    
def config_panel():
    config = {
        "dataset": None,
        "start_index": 0,
        "end_index": 0,
        "img_dir": ""  # Initialize img_dir
    }

    st.sidebar.title("SLAKE")
    dataset = load_data()  # Load the dataset
    if dataset is None:
        st.error("Dataset could not be loaded. Please try again.")
        return config  # Return the initial config if dataset is None

    split = "train"  # Assuming you want to use the 'train' split
    if split in dataset:
        dataset = dataset[split]
    else:
        st.error("Dataset does not contain the specified split. Please try again.")
        return config  # Return the initial config if split is not found

    img_dir = st.sidebar.text_input("Enter Image Folder Path", "../../images/slake")  # Default folder path
    config["img_dir"] = os.path.abspath(img_dir)  # Update img_dir in config

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

    return config  # Return the updated config dictionary

# Streamlit app
def process_dataset():
    config = config_panel()
    
    dataset = config["dataset"]

    if dataset is None:  # Check if dataset is None
        st.error("Dataset could not be loaded. Please try again.")
        return  # Exit the function if dataset is not loaded
    
    start_index = config["start_index"]
    end_index = config["end_index"]
    img_dir = config["img_dir"]  #

    for i in range(start_index, end_index):
        process_question(dataset[i], img_dir, i + 1)
    
if __name__ == "__main__":
    process_dataset()

