import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO

# Load the dataset
@st.cache_data()
def load_data():
    dataset = load_dataset("mlfoundations/VisIT-Bench", split="test")
    return dataset

# Function to load image from a URL
def load_image_from_url(image_url):
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))

# Streamlit app
def main():
    st.title("VisIT-Bench Dataset")

    # Load dataset
    dataset = load_data()

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
    page_data = dataset.select(range(start_index, end_index))

    for i in range(len(page_data)):
        row = page_data[i]
        st.header(f"Question: {start_index + i + 1}")

        # Display instruction
        st.write("**Instruction:**")
        st.write(row['instruction'])

        # Display image
        st.write("**Image:**")
        if row['image'] is not None:
            image = row['image']
            st.image(image, caption=f"Item {start_index + i + 1}", use_column_width=True)

        st.divider()

if __name__ == "__main__":
    main()

