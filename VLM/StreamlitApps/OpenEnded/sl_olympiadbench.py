import streamlit as st
from datasets import load_dataset
from vlm_chat import LlavaChat
import time 

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    ds = load_dataset("lmms-lab/OlympiadBench")
    ds = ds[split]
    return ds

def load_vlm_model():
    vlm = LlavaChat()
    return vlm

def build_prompt(question, options=None):
    if not options:  # Check if options are empty
        prompt = f"You are an expert in mathematics. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are an expert in mathematics. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
    return prompt

# Streamlit app
def main():
    
    # Sidebar for subject selection
    st.sidebar.title("OlympiadBench")
    # Split selection
    split = st.sidebar.selectbox("Select Split", ["test_en", "test_cn"])

    # Load dataset
    dataset = load_data(split)

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
        st.write(row['question'])

        # Display images
        if row['images']:
            for image in row['images']:
                try:
                    st.image(image, caption=f"Item {i + 1}", width=500)
                except Exception as e:
                    st.write(f"Unable to load image from {url}: {e}")

        vlm = load_vlm_model()
        prompt = build_prompt(question)

        # Use a button to trigger the answer retrieval
        if st.button(f"Ask VLM : {i+1}"):
            st.session_state.processing = True  # Set processing state
            with st.spinner("Retrieving answer..."):
                start_time = time.time()  # Start the timer
                try:  # Fixed indentation
                    print(prompt)  # Indented this line
                    answer = vlm.get_answer(prompt, image)  
                    elapsed_time = time.time() - start_time 
                    st.write(f"Model answer: {answer}")
                    st.write(f"Elapsed time: {elapsed_time:.3f} seconds")  # Display elapsed time
                except Exception as e:
                    st.error(f"Error retrieving answer: {str(e)}")
                finally:
                    st.session_state.processing = False  # Reset processing state

        st.divider()

if __name__ == "__main__":
    main()

