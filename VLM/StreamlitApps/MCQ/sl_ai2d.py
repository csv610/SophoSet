import streamlit as st
from datasets import load_dataset

from vlm_chat import LlavaChat
from io import BytesIO
import time 

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    """
    Load the specified subset and split of the AI2 ARC dataset.

    Args:
        subset (str): The subset of the dataset to load ('ARC-Challenge' or 'ARC-Easy').
        split (str): The split of the dataset to load ('train', 'validation', or 'test').

    Returns:
        Dataset: The specified subset and split of the dataset.
    """

    model_name = "lmms-lab/ai2d"

    try:
       ds = load_dataset("lmms-lab/ai2d")
       return ds[split]
    except Exception as e:
       st.error(f"Error loading dataset: {str(e)}")
       return None
    

def config_panel():
    """
    Configure the sidebar panel for user interaction, including subset, split, and pagination settings.

    Returns:
        tuple: Contains the loaded dataset, number of items per page, and the selected page number.
    """
    st.sidebar.title("AI2D")

    # Split selection
    split = 'test'

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Load dataset
    dataset = load_data(split)

    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items + num_items_per_page - 1) // num_items_per_page

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    return dataset, num_items_per_page, selected_page

def load_vlm_model():
    vlm = LlavaChat()
    return vlm

def build_prompt(question, options=None):
    if not options:  # Check if options are empty
        prompt = f"You are an expert in mathematics. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are an expert in mathematics. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
    return prompt

def ask_vlm(question, options, image, index):
    if st.button(f"Ask VLM : {index}"):
        vlm = load_vlm_model()
        prompt = build_prompt(question, options)

        # Convert image to bytes if it exists
        if image:
            
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='JPEG')  # Save as JPEG
            img_byte_arr.seek(0)  # Move to the beginning of the byte stream
            image = img_byte_arr  # Update image to byte stream

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

# Streamlit app
def view_dataset():

    # Get sidebar selections
    dataset, num_items_per_page, selected_page = config_panel()

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, len(dataset))

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {i + 1}")

        # Display question and answer
        question  = row['question']
        st.write(question)
        st.write("")  # Add vertical space

        image = row['image']
        st.image(image, caption=f"Qs:{i}")
        st.write("")  # Add vertical space

        options = row['options']
        # Format options as (A) option1, (B) option2, ...
        formatted_options = ', '.join([f"({chr(65 + i)}) {option}" for i, option in enumerate(options)])
    
        # Display each option on a separate line
        for option in formatted_options.split(', '):
            st.write(option)
        st.write("")  # Add vertical space

        answer = row['answer']
        
        # Add a button to reveal the answer
        if st.button(f"Correct Answer: {i + 1}"):
            st.write(f"Correct Answer: {answer}")

        ask_vlm(question, options, image, i+1)

        st.divider()

if __name__ == "__main__":
    view_dataset()
