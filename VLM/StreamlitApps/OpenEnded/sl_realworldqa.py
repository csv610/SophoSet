import streamlit as st
from datasets import load_dataset
import re
from typing import Tuple, List, Optional
from vlm_chat import LlavaChat
import time 

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
    
def load_vlm_model():
    vlm = LlavaChat()
    return vlm

def build_prompt(question, options):
    if not options:  # Check if options are empty
        prompt = f"You are an intelligent assistant. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are an intellgent assistant. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
    return prompt

def config_panel() -> Tuple[Optional[dict], int, int]:
    """Handle the contents of the left panel (sidebar)."""
    st.sidebar.title("RealworldQA")

    # Load dataset
    dataset = load_data()
    if dataset is None:
        return None, 0, 0

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + (1 if total_items % num_items_per_page != 0 else 0)

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    return dataset, num_items_per_page, selected_page

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

def display_question(index: int, row: dict):
    """Display a single question with its image and answer."""
    st.header(f"Question: {index}")

    question, options = split_text(row['question'])

    st.write(question)

    image = None
    if row['image']:
        image = row['image']
        st.image(image, caption=f"Qs:{index}")
                
    # Convert image to bytes if it exists
    if image:
        from io import BytesIO
        img_byte_arr = BytesIO()
        # Convert image to RGB mode if it is in RGBA
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(img_byte_arr, format='JPEG')  # Change 'JPG' to 'JPEG'
        img_byte_arr.seek(0)  # Move to the beginning of the byte stream
        image = img_byte_arr  # Update image to byte stream
    
    if options:
        for option in options:
            st.write(option)
    
    with st.expander("Show Answer"):
        st.write(f"Answer: {row['answer']}")
    
    vlm = load_vlm_model()
    prompt = build_prompt(question, options)

    # Use a button to trigger the answer retrieval
    if st.button(f"Ask VLM : {index}"):
        st.session_state.processing = True  # Set processing state
        with st.spinner("Retrieving answer..."):
            start_time = time.time()  # Start the timer
            try:  # Fixed indentation
                print(prompt)
                answer = vlm.get_answer(image, prompt)  
                elapsed_time = time.time() - start_time 
                st.write(f"Model answer: {answer}")
                st.write(f"Elapsed time: {elapsed_time:.3f} seconds")  # Display elapsed time
            except Exception as e:
                st.error(f"Error retrieving answer: {str(e)}")
            finally:
                st.session_state.processing = False  # Reset processing state

    st.divider()

def view_dataset():


    dataset, num_items_per_page, selected_page = config_panel()

    if dataset is None:
        return

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, len(dataset))

    for i in range(start_index, end_index):
        display_question(i + 1, dataset[i])

    # Add navigation buttons
    if selected_page > 1:
        if st.button("Previous Page"):
            st.session_state.page = selected_page - 1
            st.experimental_rerun()

    if selected_page < (len(dataset) // num_items_per_page) + 1:
        if st.button("Next Page"):
            st.session_state.page = selected_page + 1
            st.experimental_rerun()

if __name__ == "__main__":
    view_dataset()
