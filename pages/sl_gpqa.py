import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import re

hf_token = "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"

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(subset, split):
    model_name = "Idavidrein/gpqa"
    ds = load_dataset(model_name, subset,  use_auth_token=hf_token)
    return ds[split]

def rewrite_sentence(text):
    # First, identify and escape LaTeX dollar signs
    text = re.sub(r'(?<!\\)\$(.*?)(?<!\\)\$', r'\\$\1\\$', text)

    # Then, identify and ensure money dollar signs are not escaped
    text = re.sub(r'\\\$(\d+(\.\d{1,2})?)\\\$', r'$\1$', text)

    return text

# Streamlit app
def main():
    st.title("GPQA  Dataset")

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Subset selection
    subset = st.sidebar.selectbox("Select Subset", ["gpqa_diamond", "gpqa_experts", "gpqa_extended", "gpqa_main"])

    # Split selection
    split = "train"

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
        question = row['Pre-Revision Question']
        answer   = row['Pre-Revision Correct Answer']

        st.write(question)
        st.write(f"Answer: {answer}")

        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    main()

