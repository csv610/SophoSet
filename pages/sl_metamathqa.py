import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO

# Load the dataset
@st.cache_data()
def load_data():
    ds = load_dataset("meta-math/MetaMathQA")
    return ds

# Function to load image from a URL
def load_image_from_url(image_url):
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))

# Streamlit app
def main():
    st.title("MetaMathQA Dataset")

    # Load dataset
    dataset = load_data()
    dataset = dataset['train']  # You can also choose 'test', 'validation', etc.

    # Sidebar for navigation
    st.sidebar.title("Navigation")

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
        st.header(f"Question {start_index + i + 1}")
        st.write(row['query'])
        st.write(f"Answer: {row['response']}")

        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    main()

