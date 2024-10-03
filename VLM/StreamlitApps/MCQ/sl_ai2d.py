import time
from io import BytesIO

import streamlit as st
from datasets import load_dataset
from vlm_chat import LlavaModel

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    model_name = "lmms-lab/ai2d"
    try:
       ds = load_dataset(model_name)
       return ds[split]
    except Exception as e:
       st.error(f"Error loading dataset: {str(e)}")
       return None
    
def config_panel():
    st.sidebar.title("AI2D")

    # Initialize the dictionary to hold the return values
    config = {
        "dataset": None,
        "start_index": 0,
        "end_index": 0
    }

    dataset = load_data('test')

    if dataset:
        total_items = len(dataset)
        num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
        total_pages = (total_items + num_items_per_page - 1) // num_items_per_page

        page_options = list(range(1, total_pages + 1))
        selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

        start_index = (selected_page - 1) * num_items_per_page
        end_index = min(start_index + num_items_per_page, total_items)

        # Update the config dictionary with actual values
        config["dataset"] = dataset
        config["start_index"] = start_index
        config["end_index"] = end_index

    return config 

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

def ask_vlm(params):
    question = params["question"]
    options  = params["options"]
    image    = params["image"]
    index    = params["id"]

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

def process_question(row, qindex):
    st.header(f"Question: {qindex}")

    # Display question and answer
    question = row['question']
    st.write(question)
    st.write("")  # Add vertical space

    image = row['image']
    if image:
        st.image(image, caption=f"Qs:{qindex}")
        st.write("")  # Add vertical space

    options = row['options']
    # Format options as (A) option1, (B) option2, ...
    formatted_options = ', '.join([f"({chr(65 + i)}) {option}" for i, option in enumerate(options)])
    
    # Display each option on a separate line
    for option in formatted_options.split(', '):
        st.write(option)
    st.write("")  # Add vertical space
    
    # Add a button to reveal the answer
    if st.button(f"Correct Answer #{qindex}"):
        answer = row['answer']
        st.write(f"Correct Answer: {answer}")

    # Create a dictionary to hold the parameters
    vlm_params = {
        "question": question,
        "options": options,
        "image": image,
        "id": qindex
    }
    
    ask_vlm(vlm_params)  # Pass the dictionary to the function

    st.divider()

def process_dataset():

    config = config_panel()

    dataset = config["dataset"]
   
    if dataset is None:
        st.error("Dataset could not be loaded. Please try again.")
        return
    
    start_index = config["start_index"]
    end_index = config["end_index"]

    for i in range(start_index, end_index):
        process_question(dataset[i], i+1)

if __name__ == "__main__":
    process_dataset()
