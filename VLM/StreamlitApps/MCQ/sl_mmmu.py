import time
import re
from io import BytesIO

import streamlit as st
from datasets import load_dataset
from vlm_chat import LlavaModel

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(subset, split):
    """
    Load the specified subset and split of the AI2 ARC dataset.

    Args:
        subset (str): The subset of the dataset to load.
        split (str): The split of the dataset to load ('train', 'validation', or 'test').

    Returns:
        Dataset: The specified subset and split of the dataset.

    Raises:
        ValueError: If the dataset loading fails.
    """
    try:
        ds = load_dataset("MMMU/MMMU", subset)
        if split not in ds:
            raise ValueError(f"Invalid split '{split}' for subset '{subset}'")
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        raise ValueError(f"Failed to load dataset for subset '{subset}' and split '{split}'")

def config_panel():
    st.sidebar.title("MMMU")

    # Initialize the dictionary to hold the return values
    config = {
        "dataset": None,
        "subject": None,  # Initialize subject
        "start_index": 0,
        "end_index": 0
    }

    # Subset selection
    subset = st.sidebar.selectbox("Select Subset", ["Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory", "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science", "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics", "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage", "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology"])

    # Split selection
    split = st.sidebar.selectbox("Select Split", ["test", "validation", "dev"])

    # Load dataset
    config["dataset"] = load_data(subset, split)

    if config["dataset"]:
        config["subject"] = subset  # Set the subject based on the selected subset
        num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
        total_items = len(config["dataset"])
        total_pages = (total_items + num_items_per_page - 1) // num_items_per_page

        selected_page = st.sidebar.selectbox("Select Page Number", options=list(range(1, total_pages + 1)))

        # Calculate start and end index
        start_index = (selected_page - 1) * num_items_per_page
        end_index = min(start_index + num_items_per_page, total_items)

        # Update the config dictionary with actual values
        config["start_index"] = start_index
        config["end_index"] = end_index

    return config  # Return the config dictionary

@st.cache_resource
def load_vlm_model():
    vlm = LlavaModel()
    return vlm

def build_prompt(question, subject, options=None):

    if not options:  # Check if options are empty
        prompt = f"You are an expert in {subject}. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are an expert in {subject}. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
    return prompt

def ask_vlm(params):
    question = params["question"]
    subject = params["subject"]
    options = params["options"]
    images = params["images"] 
    index = params["id"]

    if st.button(f"Ask VLM #{index}"):
        vlm = load_vlm_model()
        prompt = build_prompt(question, subject, options)

        st.session_state.processing = True  # Set processing state
        with st.spinner("Retrieving answer..."):
            start_time = time.time()  # Start the timer
            try:
                answer = vlm.get_answer(prompt, images)  # Pass the images list
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                st.write(f"Model answer: {answer}")
                st.write(f"Elapsed time: {elapsed_time:.3f} seconds")  # Display elapsed time
            except Exception as e:
                st.error(f"Error retrieving answer: {str(e)}")
            finally:
                st.session_state.processing = False  # Reset processing state

def parse_options(options_str):
    # Remove the outer square brackets
    options_str = options_str.strip('[]')
    
    # Split the string by commas, but not within quotes
    pattern = r',\s*(?=[\'"])'
    options = re.split(pattern, options_str)
    
    # Clean up each option
    cleaned_options = []
    for option in options:
        # Remove surrounding quotes and whitespace
        cleaned = option.strip().strip('\'"')
        # Ensure dollar signs are preserved
        if cleaned.startswith('$'):
            cleaned = '$' + cleaned.lstrip('$')
        cleaned_options.append(cleaned)
    
    return cleaned_options

def process_question(row, subject, index):
    st.header(f"Question #{index}")

    # Display question 
    question = row['question']
    st.write(question)
    st.write("")  # Add vertical space

    images = []
    for j in range(1, 8):
        image_key = f"image_{j}"
        if row[image_key] is not None:
            image = row[image_key]
            if image:
                images.append(image)
                st.image(image, caption=f"Qs: {index} Image{j}")
    st.write("")  # Add vertical space

    options_str = row['options']
    options = parse_options(options_str)

    if options:
        st.write("Options:")
        for idx, option in enumerate(options):
            st.write(f"({chr(65 + idx)}) {option}")

    if st.button(f"Show Correct Answer #{index}"):
        st.write(row['answer'])
        st.write("") 

    # Create a dictionary to hold the parameters
    vlm_params = {
        "question": question,
        "subject": subject,
        "options": options,
        "images": images,  
        "id": index
    }
    
    ask_vlm(vlm_params)  # Pass the dictionary to the function
    
    st.divider()

def process_dataset():

    config = config_panel() 

    dataset = config["dataset"]
    
    if dataset is None:
        st.error("Dataset could not be loaded. Please try again.")
        return
    
    subject = config["subject"] 
    start_index = config["start_index"]
    end_index = config["end_index"]

    for i in range(start_index, end_index):
        process_question(dataset[i], subject, i+1)  

if __name__ == "__main__":
    process_dataset()

