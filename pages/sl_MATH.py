import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import re

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(subset, split):
    model_id = "lighteval/MATH"
    try:
       ds = load_dataset(model_id, subset)
       return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Streamlit app
def view_dataset():
    st.title("MATH Dataset")

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Subset selection
    subset = st.sidebar.selectbox("Select Subset", [
        "all",
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus"
    ])

    # Split selection
    split = st.sidebar.selectbox("Select Split", ["train", "test"])

    # Load dataset
    dataset = load_data(subset, split)

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
        st.header(f"Question: {start_index + i + 1}")

        # Display question and answer
        problem = row['problem']
        solution = row['solution']

        st.write(problem)
        st.write("")
        st.write(f"Solution: {solution}")

        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    view_dataset()

