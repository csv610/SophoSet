import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(subject):
    ds = load_dataset("GAIR/OlympicArena", subject)
    return ds

# Function to load image from a URL
def load_image(image_url):
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))

# Streamlit app
def main():
    st.title("OlympicArena Dataset")
    st.divider()

    # Sidebar for subject selection
    st.sidebar.title("Navigation")
    subject = st.sidebar.selectbox(
        "Select Subject",
        options=["Astronomy", "Biology", "CS", "Chemistry", "Geography", "Math", "Physics"]
    )

    # Load dataset
    dataset = load_data(subject)

    split = "test"
    dataset = dataset[split]

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

        # Display instruction
        st.write(row['problem'])

        # Display images
        if row['figure_urls']:
            for url in row['figure_urls']:
                try:
                    image = load_image(url)
                    st.image(image, caption=f"Item {start_index + i + 1}", use_column_width=True)
                except Exception as e:
                    st.write(f"Unable to load image from {url}: {e}")

        st.divider()

if __name__ == "__main__":
    main()

