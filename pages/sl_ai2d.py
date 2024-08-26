import streamlit as st
from datasets import load_dataset
import re

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
    st.sidebar.title("Navigation")

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

# Streamlit app
def view_dataset():
    """
    Display the AI2 ARC dataset in a paginated format with questions and choices labeled as (A), (B), etc.
    """
    st.title("AI2D Dataset")
    st.divider()

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
        answer    = row['answer']
        choices   = row['options']
        image     = row['image']

        # Use the helper function to display text with custom font size and add spacing
        st.write(question)
        st.write("")  # Add vertical space
        st.write(choices)
        st.image(image, caption=f"Qs:{i}")
        st.write("")  # Add vertical space
        st.write(f"Answer: {answer}")
        st.write("")  # Add vertical space

        st.divider()

if __name__ == "__main__":
    view_dataset()

