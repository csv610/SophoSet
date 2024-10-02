import streamlit as st
from datasets import load_dataset

from vlm_chat import LlavaChat
import time  # Ensure to import time at the top of the file

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data():
    try:
        ds = load_dataset("TIGER-Lab/TheoremQA")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None  # Return None or handle as needed
    return ds['test']

@st.cache_resource()
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
                st.write(f"Model answer #{answer}")
                st.write(f"Elapsed time: {elapsed_time:.3f} seconds")  # Display elapsed time
            except Exception as e:
                st.error(f"Error retrieving answer: {str(e)}")
            finally:
                st.session_state.processing = False 
                
def config_panel():
    """Function to handle the contents of the left panel (sidebar)."""
    st.sidebar.title("TheoremQA")

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Load dataset
    dataset = load_data()

    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + 1

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, len(dataset))

    return dataset, start_index, end_index

def process_question(row, index):
    st.header(f"Question #{index}")

    # Display question and answer
    question = row['Question']
    st.write(question)
    st.write("")  # Add vertical space

    image = None
    if row['Picture']:
        image = row['Picture']
        st.image(image)  # Display the image

    if st.button(f"Show Correct Answer #{index}"):
        answer = row['Answer']
        st.write(answer)

    ask_vlm(question, None, image, index)

    st.divider()  # Divider between items


# Streamlit app
def process_dataset():   
    dataset, start_index, end_index = config_panel()
    
    if dataset is None:  # Check if dataset is None
        st.error("Dataset could not be loaded. Please try again.")
        return  # Exit the function if dataset is not loaded

    for i in range(start_index, end_index):
        process_question(dataset[i], i+1)

if __name__ == "__main__":
    process_dataset()

