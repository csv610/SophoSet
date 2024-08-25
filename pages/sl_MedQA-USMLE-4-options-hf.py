import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    try:
       ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf")
       return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Streamlit app
def main():
    st.title("MedQA-USMLE-4 Dataset")

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Split selection
    split = st.sidebar.selectbox("Select Split", ["train", "validation", "test"])

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
        st.header(f"Question: {start_index + i + 1}")

        # Display question and answer
        st.write(row['sent1'])
        st.write(f"(A)  {row['ending0']}")
        st.write(f"(B)  {row['ending1']}")
        st.write(f"(C)  {row['ending2']}")
        st.write(f"(D)  {row['ending3']}")
        st.write(f"label: {row['label']}")

        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    main()

