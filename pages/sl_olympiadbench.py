import streamlit as st
from datasets import load_datasets

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    ds = load_dataset("lmms-lab/OlympiadBench")
    ds = ds[split]
    return ds

# Streamlit app
def main():
    st.title("Dataset: OlympicArena")
    st.divider()

    # Sidebar for subject selection
    st.sidebar.title("Navigation")
    # Split selection
    split = st.sidebar.selectbox("Select Split", ["test_en", "test_cn"])

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
        st.header(f"Question: {i + 1}")

        # Display instruction
        st.write(row['question'])

        # Display images
        if row['images']:
            for image in row['images']:
                try:
                    st.image(image, caption=f"Item {i + 1}", width=500)
                except Exception as e:
                    st.write(f"Unable to load image from {url}: {e}")

        st.divider()

if __name__ == "__main__":
    main()

