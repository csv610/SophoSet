import streamlit as st
from datasets import load_dataset

st.set_page_config(layout="wide")

# Define SPLITS constant
SPLITS = ["test", "testmini"]

# Load the datasetsl
@st.cache_data()
def load_data(split):
    model_name = "qintongli/GSM-Plus"
    try:
        ds = load_dataset(model_name)
        if split not in ds:
            raise ValueError(f"Split '{split}' not found in the dataset.")
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Streamlit app
def main():
    st.title("Dataset: GSM Plus")
    st.divider()

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Split selection
    split = st.sidebar.selectbox("Select Split", SPLITS)

    # Load dataset
    dataset = load_data(split)
    if dataset is None:
        st.stop()  # Stop execution if dataset loading failed

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + 1

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    # Font size selection
    font_size = st.sidebar.slider("Select Font Size", min_value=10, max_value=40, value=20)

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, total_items)

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {start_index + i + 1}")

        question = row['question']
        answer = row['answer']

        st.write(question)
        st.write("")
        
        # Create a unique key for each button
        button_key = f"show_answer_{i}"
        if st.button("Show Answer", key=button_key):
            st.write(f"Answer: {answer}")
        
        st.divider()

if __name__ == "__main__":
    main()