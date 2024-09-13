import streamlit as st

from llm_chat import load_llm_model, ask_llm
from datasets import load_dataset

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data():
    try:
        ds = load_dataset("theblackcat102/IMO-geometry")
        return ds['test']
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None
    
def config_panel():
    """Function to handle the contents of the left panel (sidebar)."""
    st.sidebar.title("Navigation")

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Load dataset
    dataset = load_data()

    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + 1

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    return dataset, num_items_per_page, selected_page

# Streamlit app
def view_dataset():
    st.title("Dataset: IMO-Geometry")
    st.divider()

    # Get sidebar selections
    dataset, num_items_per_page, selected_page = config_panel()

    if dataset is None:
        st.warning("Unable to load dataset. Please try again later.")
        return

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, len(dataset))

    model_name = "llama3.1"
    llm = load_llm_model(model_name)
    for i in range(start_index, end_index):
        row = dataset[i]

        question = row['question']

        st.header(f"Question: {i + 1}")
        st.write(question)

        ask_llm(llm, question, i+1)

        st.divider()  # Divider between items

if __name__ == "__main__":
    view_dataset()