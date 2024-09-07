import streamlit as st
from datasets import load_dataset
import re

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(subset, split):
    try:
        model_name = "openai/gsm8k"
        ds = load_dataset(model_name, subset)
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def rewrite_sentence(text):
    # First, identify and escape LaTeX dollar signs
    text = re.sub(r'(?<!\\)\$(.*?)(?<!\\)\$', r'\\$\1\\$', text)

    # Then, identify and ensure money dollar signs are not escaped
    text = re.sub(r'\\\$(\d+(\.\d{1,2})?)\\\$', r'$\1$', text)

    return text

def split_solution_answer(text):
    """Function to split the answer text into solution and answer."""
    parts = text.split('####')
    if len(parts) == 2:
        solution = parts[0].strip()
        answer = parts[1].strip()
        return answer, solution
    return text, ""  # If '####' is not found, return the original text and empty string


def config_panel():
    """Function to handle the contents of the left panel (sidebar)."""
    st.sidebar.title("Navigation")

    # Subset selection
    subset = st.sidebar.selectbox("Select Subset", ["main", "socratic"])

    # Split selection
    split = st.sidebar.selectbox("Select Split", ["train", "test"])


    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Load dataset
    dataset = load_data(subset, split)

    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + 1

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    return dataset, num_items_per_page, selected_page

# Streamlit app
def view_dataset():
    st.title("Dataset: GSM8K")
    st.divider()

    # Get sidebar selections
    dataset, num_items_per_page, selected_page = config_panel()
    
    if dataset is None:
        st.warning("Unable to load dataset. Please check your internet connection and try again.")
        return

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, len(dataset))

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {i + 1}")

        # Display question
        question = rewrite_sentence(row['question'])
        st.write(question)
        st.write("")  # Add vertical space

        # Split the answer into solution and the final answer
        raw_answer = rewrite_sentence(row['answer'])
        answer, solution = split_solution_answer(raw_answer)

        # Add buttons for showing answer and solution
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Show Answer {i}"):
                st.write(f"Answer: {answer}")
        with col2:
            if st.button(f"Show Solution {i}"):
                st.write(f"Solution: {solution}")

        st.divider()  # Divider between items

if __name__ == "__main__":
    view_dataset()