import streamlit as st
import re

from  llm_chat import LLMChat
from datasets import load_dataset

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    model_name = "qwedsacf/competition_math"
    ds = load_dataset(model_name)
    return ds[split]

@st.cache_resource
def load_llm_model(model_name: str = "llama3.1"):
    """Initialize and cache the LLM."""
    return LLMChat(model_name)

def rewrite_sentence(text):
    # First, identify and escape LaTeX dollar signs
    text = re.sub(r'(?<!\\)\$(.*?)(?<!\\)\$', r'\\$\1\\$', text)

    # Then, identify and ensure money dollar signs are not escaped
    text = re.sub(r'\\\$(\d+(\.\d{1,2})?)\\\$', r'$\1$', text)

    return text

# Streamlit app
def main():
    st.title("Dataset: Competitive Math")
    st.divider()

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Split selection
    split = "train"

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

    llm = load_llm_model()

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {i + 1}")

        # Display question and answer
        problem = rewrite_sentence(row['problem'])

        st.write(problem)

        if st.button(f"Human Answer :{i+1}"):
           st.write(f"Solution: {row['solution']}")

        if st.button(f"LLM Answer :{i+1}"):
           with st.spinner("Processing ..."):
                response = llm.get_answer(problem)
                st.write(f"Answer: {response['answer']}")

        st.divider()

if __name__ == "__main__":
    main()

