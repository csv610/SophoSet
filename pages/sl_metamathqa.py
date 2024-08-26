import streamlit as st
from datasets import load_dataset

# Load the dataset
@st.cache_data()
def load_data(split='train'):
    try:
        ds = load_dataset("meta-math/MetaMathQA")
        if split not in ds:
            raise ValueError(f"Invalid split: {split}. Available splits are: {', '.join(ds.keys())}")
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Streamlit app
def main():
    st.title("Dataset: MetaMathQA")
    st.divider()

    # Load dataset
    dataset = load_data()
    if dataset is None:
        st.stop()

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
        st.header(f"Question {i + 1}")
        st.write(row['query'])
        st.write(f"Answer: {row['response']}")

        st.divider()  # Divider between items

if __name__ == "__main__":
    main()
