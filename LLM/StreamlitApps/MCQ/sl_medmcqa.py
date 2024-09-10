import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import re

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    ds = load_dataset("openlifescienceai/medmcqa")
    return ds[split]

# Streamlit app
def main():
    st.title("Medmcqa Dataset")

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
        question = row['question']
        opa = row['opa']
        opb = row['opb']
        opc = row['opc']
        opd = row['opd']
        cop = row['cop']
        exp = row['exp']

        st.write(question)
        st.write(f"(A): {opa}")
        st.write(f"(B): {opb}")
        st.write(f"(C): {opc}")
        st.write(f"(D): {opd}")
        st.write(f"Answer: {cop}")
        st.write(f"Explanation: {exp}")


        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    main()

