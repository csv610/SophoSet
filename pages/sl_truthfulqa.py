import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(subset, split):
    ds = load_dataset("truthfulqa/truthful_qa", subset)
    return ds[split]

# Streamlit app
def main():
    st.title("TruthfulQA Dataset")

    # List of available subjects
    subsets = ['generation', 'multiple_choice']
    
    # Sidebar for subject and split selection
    subset = st.sidebar.selectbox("Select Subject", subsets)
    split = st.sidebar.selectbox("Select Split", ["validation"])
    
    # Load dataset
    dataset = load_data(subset, split)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + (total_items % num_items_per_page > 0)

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
        st.write(f"**Question:** {row['question']}")

        if subset == "generation":
            st.write("**Correct Answers:**")
            for answer in row['correct_answers']:
                st.write(f"- {answer}")
            
            st.write("**Best Answer:**")
            st.write(row['best_answer'])

            st.write("**Incorrect Answers:**")
            for answer in row['incorrect_answers']:
                st.write(f"- {answer}")

        if subset == "multiple_choice":
            st.write("**mc1 targets:**")
            for choice in row['mc1_targets']['choices']:
                st.write(f"- {choice}")
            st.write(f"labels:{row['mc1_targets']['labels']}")

            st.write("**mc2 targets:**")
            for choice in row['mc2_targets']['choices']:
                st.write(f"- {choice}")
            st.write(f"labels: {row['mc2_targets']['labels']}")

        st.divider()

if __name__ == "__main__":
    main()

