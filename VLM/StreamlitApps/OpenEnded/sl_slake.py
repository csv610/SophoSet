import streamlit as st
from datasets import load_dataset
from PIL import Image
import os

from vlm_chat import LlavaChat
from io import BytesIO
import time 

# Load the dataset
@st.cache_data()
def load_data():
    ds = load_dataset("BoKelvin/SLAKE")
    return ds

# Function to load image from a local path
def load_image_from_path(image_path):
    return Image.open(image_path)

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

# Convert image to bytes if it exists
        if image:
            from io import BytesIO
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
def main():
    # Load dataset
    dataset = load_data()

    split = "train"
    dataset = dataset[split]

    # Sidebar for navigation
    st.sidebar.title("SLAKE")

    # Ask for image folder
    img_dir = st.sidebar.text_input("Enter Image Folder Path", "../../images/slake")  # Default folder path
    img_dir = os.path.abspath(img_dir)  # Convert to absolute path

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + 1

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, total_items)

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {i + 1}")

        # Display instruction
        question = row['question']
        st.write(question)
  
        image = None
        if row['img_name']:
            image_path = os.path.join(img_dir, row['img_name'])
            if os.path.exists(image_path):
                image = load_image_from_path(image_path)
                st.image(image, caption=f"Qs: {i + 1}", use_column_width=True)
            else:
                st.write("Image file not found.")

        if st.button(f"Correct Answer: {i + 1}"):
            st.write(row['answer'])

        ask_vlm(question, None, image, i+1)
    
        st.divider()

if __name__ == "__main__":
    main()

