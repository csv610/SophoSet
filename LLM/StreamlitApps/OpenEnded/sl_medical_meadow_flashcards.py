import re
import streamlit as st

from llm_chat import load_llm_model, ask_llm
from datasets import load_dataset

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    model_name = "medalpaca/medical_meadow_medical_flashcards"
    try:
        ds = load_dataset(model_name)
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None
    
# Streamlit app
def main():
    st.title("Dataset: Medical Meadow Flashcards")
    st.divider()

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Split selection
    split = "train"

    # Load dataset
    dataset = load_data(split)

    if dataset is None:
        st.warning("Unable to load dataset. Please try again later.")
        return

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

    model_name =  "llama3.1"
    llm = load_llm_model(model_name)

    for i in range(start_index, end_index):
        row = dataset[i]

        question = row['input']

        st.header(f"Question: {i + 1}")
        st.write(question)
        st.write(" ")

        if st.button(f"Human Answer:{i+1}"):
            answer  = row['output']
            st.write(f"Answer: {answer}")
        
        ask_llm(llm, question, i+1)

        st.divider()

if __name__ == "__main__":
    main()
