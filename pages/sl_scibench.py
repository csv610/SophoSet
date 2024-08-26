import streamlit as st
from datasets import load_dataset

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split = 'train'):
    ds = load_dataset("xw27/scibench")
    ds = ds[split]
    return ds

# Streamlit app
def main():
    st.title("Dataset: Sci-Bench")
    st.divider()

    # Load dataset
    dataset = load_data()

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items - 1) // num_items_per_page + 1  # Simplified calculation

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, total_items)

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question {i + 1}")
        st.write(row['problem_text'])
        st.write(f"Answer: {row['answer_number']}")
        st.divider()

if __name__ == "__main__":
    main()
