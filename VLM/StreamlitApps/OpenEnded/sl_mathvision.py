
import time  

import streamlit as st
from datasets import load_dataset
from vlm_chat import LlavaModel

# Constants
MODEL_NAME = "MathLLMs/MathVision"
DEFAULT_NUM_QUESTIONS = 5

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split: str):
    model_name = MODEL_NAME
    try:
        ds = load_dataset(model_name)
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None
    
def config_panel():

    # Add a session state variable to track the processing state
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Sidebar for navigation
    st.sidebar.title("MathVision")
    
    # Select subset
    split = st.sidebar.selectbox("Select Dataset Subset", options=['test', 'testmini'])

    param = {
        "dataset": None,
        "start_index": 0,
        "end_index": 0
    }

    # Load dataset
    param["dataset"] = load_data(split)  # Store the loaded dataset in the param dictionary
    
    if param["dataset"] is None:
        st.warning("Unable to load dataset. Please try again later.")
        return config  # Return the initialized dictionary

    # Select number of questions per page
    num_questions_per_page = st.sidebar.slider("Select Number of Questions per Page", min_value=1, max_value=10, value=DEFAULT_NUM_QUESTIONS)
    
    # Calculate total pages
    total_questions = len(param["dataset"])
    total_pages = (total_questions // num_questions_per_page) + 1

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    # Display questions for the selected page
    start_index = (selected_page - 1) * num_questions_per_page
    end_index   = min(start_index + num_questions_per_page, total_questions)

    # Cancel button to stop processing
    if st.session_state.processing and st.sidebar.button("Cancel"):
        st.session_state.processing = False  # Logic to cancel the process
        st.warning("Processing canceled.")

    param['start_index'] = start_index
    param['end_index']   = end_index

    return param

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

def ask_vlm(question: str, options: list, image, index: int):
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

def process_question(row, index):
    st.header(f"Question #{index}")

    question = row['question']
    st.write(question)

    image = None
    if row['decoded_image']:
        image = row['decoded_image']
        st.image(image, caption=f"Question {index}", use_column_width=True)

    options = None
    if row['options'] and len(row['options']) > 0:
        options = row['options']
        st.write(options)

    if row['answer']:
        if st.button(f"Correct Answer #{index}"):
            st.write(row['answer'])
        
    ask_vlm(question, options, image, index)

    st.divider()
  
# Streamlit app
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
