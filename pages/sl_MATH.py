import streamlit as st
from datasets import load_dataset

st.set_page_config(layout="wide")

# Constants
MODEL_ID = "lighteval/MATH"

SUBSETS = [
    "all",
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus"
]
SPLITS = ["train", "test"]

# Load the dataset
@st.cache_data()
def load_data(subset: str, split: str):
    try:
       ds = load_dataset(MODEL_ID, subset)
       return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Streamlit app
def view_dataset():
    st.title("Dataset: MATH")
    st.divider()

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Subset selection
    subset = st.sidebar.selectbox("Select subset", SUBSETS)

    # Split selection
    split = st.sidebar.selectbox("Select split", SPLITS)

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
        st.header(f"Question: {i + 1}")

        # Display question and answer
        problem = row['problem']
        solution = row['solution']

        st.write(problem)
        st.write("")
        st.write(f"Solution: {solution}")

        st.divider()

if __name__ == "__main__":
    view_dataset()
