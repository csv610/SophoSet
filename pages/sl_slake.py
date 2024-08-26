import streamlit as st
from datasets import load_dataset
from PIL import Image
import os

# Load the dataset
@st.cache_data()
def load_data():
    ds = load_dataset("BoKelvin/SLAKE")
    return ds

# Function to load image from a local path
def load_image_from_path(image_path):
    return Image.open(image_path)

# Streamlit app
def main():
    st.title("Dataset: SLAKE")
    st.divider()

    # Load dataset
    dataset = load_data()

    split = "train"
    dataset = dataset[split]

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

    img_dir = os.path.abspath("slakeimages")
    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {i + 1}")

        # Display instruction
        st.write(row['question'])
  
        # Display image
        if row['img_name']:
            image_path = os.path.join(img_dir, row['img_name'])
            if os.path.exists(image_path):
                image = load_image_from_path(image_path)
                st.image(image, caption=f"Qs: {start_index + i + 1}", use_column_width=True)
            else:
                st.write("Image file not found.")
        

        st.write(f"Answer: {row['answer']}")

        st.divider()

if __name__ == "__main__":
    main()

