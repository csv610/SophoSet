import streamlit as st
from datasets import load_dataset
from llm_chat import load_llm_model, ask_llm

# Load the dataset
@st.cache_data()
def load_data(split):
    ds = load_dataset("automated-research-group/winogrande")
    return ds[split]

# Streamlit app
def main():
    st.title("Dataset: Winogrande")
    st.divider()

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Load dataset
    split = "validation"
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

    model_name = "llama3.1"
    llm = load_llm_model(model_name)

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {i + 1}")

        # Display question and answer
        st.write(row['request'])
        
        if st.button(f"Human Answer:{i+1}"):
            st.write(f"response: {row['response']}")

        st.divider()

if __name__ == "__main__":
    main()
