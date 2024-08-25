import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import string

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    return ds[split]

# Function to load image from a URL
def load_image_from_url(image_url):
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))

# Streamlit app
def main():
    st.title("MMLU Pro Dataset")

    # List of available subjects
    split = st.sidebar.selectbox("Select Split", ["test", "validation"])
    
    # Load dataset
    dataset = load_data(split)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + (total_items % num_items_per_page > 0)

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, total_items)

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {i + 1}")

        # Display instruction
        st.write(row['question'])

        # Display choices with (A), (B), (C), etc.
        choices = row['options']
        choice_labels = list(string.ascii_uppercase)[:len(choices)]  # Generate labels from A to Z
        formatted_choices = [f"({label}) {choice}" for label, choice in zip(choice_labels, choices)]
        
        st.write("**Choices:**")
        for choice in formatted_choices:
            st.write(choice)

        st.write(f"Answer: {row['answer']}")
    
        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    main()

