import streamlit as st
from datasets import load_dataset
import re
from typing import Tuple, List, Optional

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

def config_panel() -> Tuple[Optional[dict], int, int]:
    """Handle the contents of the left panel (sidebar)."""
    st.sidebar.title("Navigation")

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
    st.image(row['image'], caption=f"Question {index}")
    
    if options:
        for option in options:
            st.write(option)
    
    with st.expander("Show Answer"):
        st.write(f"Answer: {row['answer']}")

    st.divider()

def view_dataset():
    """Main function to display the dataset viewer."""
    st.title("RealWorld Dataset")

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
