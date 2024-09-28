import streamlit as st
from datasets import load_dataset
import string

from vlm_chat import LlavaChat
from io import BytesIO
import time 

st.set_page_config(layout="wide")

# Define subsets
SUBSETS = [
    'Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 'Relative_Reflectance',
    'Visual_Correspondence', 'Counting', 'IQ_Test', 'Object_Localization',
    'Semantic_Correspondence', 'Visual_Similarity', 'Forensic_Detection', 'Jigsaw',
    'Relative_Depth', 'Spatial_Relation'
]

# Define splits
SPLITS = ["val", "test"]

# Load the dataset
@st.cache_data()
def load_data(subset, split):
    model_name = "BLINK-Benchmark/BLINK"

    try:
        ds = load_dataset(model_name, subset)
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None
    
def load_vlm_model():
    vlm = LlavaChat()
    return vlm

def build_prompt(question, options=None):
    if not options:  # Check if options are empty
        prompt = f"You are a helpful assistant. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are a helpful assistant. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
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
def main():
    # Sidebar for navigation
    st.sidebar.title("BLINK")

    # Subset selection
    subset = st.sidebar.selectbox("Select Subset", SUBSETS)

    # Split selection
    split = st.sidebar.selectbox("Select Split", SPLITS)

    # Load dataset
    dataset = load_data(subset, split)

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + (1 if total_items % num_items_per_page != 0 else 0)

    # Page selection box
    selected_page = st.sidebar.selectbox("Select Page Number", options=range(1, total_pages + 1))

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, total_items)

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question {i + 1}")

        # Display question
        question = row['question']
        st.write(question)

        # Display images
        images = []
        for j in range(1, 5):
            image_key = f'image_{j}'
            if row[image_key] is not None:
                image = row[image_key]
                images.append(image)
                st.image(image, caption=f"Image {j}", width = 500)

        options = NotImplementedError
        if row['choices'] is not None:
            options = row['choices']
            for idx, option in enumerate(options):
                st.write(f"({chr(65 + idx)}) {option}")  # chr(65) is 'A'
    
        if st.button(f"Correct Answer: {i + 1}"):
            st.write(row['answer'])

        ask_vlm(question, options, images, i+1)

        st.divider()

if __name__ == "__main__":
    main()
