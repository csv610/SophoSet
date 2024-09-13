import streamlit as st

from llm_chat import load_llm_model, ask_llm
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

    model_name = "llama3.1"
    llm = load_llm_model(model_name)

    for i in range(start_index, end_index):
        row = dataset[i]

        problem = row['problem']

        st.header(f"Question: {i + 1}")
        st.write(problem)

        if st.button(f"Human Answer:{i+1}"):
            solution = row['solution']
            st.write(f"Solution: {solution}")
        
        ask_llm(llm, problem, i+1)

        st.divider()

if __name__ == "__main__":
    view_dataset()
