import re
import ast
import streamlit as st

from llm_chat import load_llm_model, ask_llm
from datasets import load_dataset

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    try:
        ds = load_dataset("medalpaca/medical_meadow_medqa")
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Streamlit app
def main():
    st.title("Dataset: Medical Meadow MedQA")
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

    model_name = "llama3.1"
    llm =  load_llm_model(model_name)

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {i + 1}")

        # Display question and answer
        question = row['input']
        answer = row['output']

        # Separate the question and choices
        match = re.search(r"\?\s*(\{.*\})", question)
        if match:
            question= question[:match.start()] + "?"
            choices_dict_str = match.group(1)
            choices_dict = ast.literal_eval(choices_dict_str)  # Convert string to dictionary
        else:
            choices_dict = {}

        st.write(question)
        
        if choices_dict:
            st.write("Options:")
            for key, choice in choices_dict.items():
                st.write(f"{key} {choice}")
        
        # Displaying answer with a button
        if st.button(f"Show Answer for Question {i + 1}"):
            short_answer = answer.strip()[0]
            st.write(f"Answer: {short_answer}")

        st.divider()

if __name__ == "__main__":
    main()
