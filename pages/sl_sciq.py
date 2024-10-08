import streamlit as st
from datasets import load_dataset
import random

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    ds = load_dataset("allenai/sciq")
    return ds[split]

def config_panel():
    """Function to handle the contents of the left panel (sidebar)."""
    st.sidebar.title("Navigation")

    # Split selection
    split = st.sidebar.selectbox("Select Split", ["train", "validation", "test"])

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Load dataset
    dataset = load_data(split)

    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + 1

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    return dataset, num_items_per_page, selected_page

# Streamlit app
def view_dataset():
    st.title("Dataset: SCIQ")
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
        question = row['question']
        st.write(question)
        st.write("")  # Add vertical space

        op1      = row['correct_answer']
        op2      = row['distractor1']
        op3      = row['distractor2']
        op4      = row['distractor3']
        options  = [op1, op2, op3, op4]
        random.shuffle(options)

        st.write(options)
        st.write("")  # Add vertical space

        st.write(f"Answer: {op1}")

        st.write(f"Explain: {row['support']}")

        st.divider()  # Divider between items

if __name__ == "__main__":
    view_dataset()

