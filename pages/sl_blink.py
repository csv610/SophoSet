import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import re
import string  # Import the string module

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(subset, split):
    model_name  = "BLINK-Benchmark/BLINK"

    try:
       ds = load_dataset(model_name, subset)
       return ds[split]
    except Exception as e:
       st.error(f"Error loading dataset: {str(e)}")
       return None

# Streamlit app
def main():
    st.title("BLINK Dataset")

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Subset selection
    subset = st.sidebar.selectbox("Select Subset", ['Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 'Relative_Reflectance', 'Visual_Correspondence', 'Counting', 'IQ_Test', 'Object_Localization', 'Semantic_Correspondence', 'Visual_Similarity', 'Forensic_Detection', 'Jigsaw', 'Relative_Depth', 'Spatial_Relation'])

    # Split selection
    split = st.sidebar.selectbox("Select Split", ["val", "test"])

    # Load dataset
    dataset = load_data(subset, split)

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + (1 if total_items % num_items_per_page != 0 else 0)

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, total_items)

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {start_index + i + 1}")

        # Display question
        question = row['question']
        st.write(question)

        # Display images
        for j in range(1, 5):
            image_key = f'image_{j}'
            if row[image_key] is not None:
                st.image(row[image_key], caption=f"Image {j}", use_column_width=True)

        if row['choices'] is not None:
           choices = row['choices']
           choice_labels = list(string.ascii_uppercase)[:len(choices)]  # Generate labels from A to Z
           formatted_choices = [f"({label}) {choice}" for label, choice in zip(choice_labels, choices)]
        
           st.write("**Choices:**")
           for choice in formatted_choices:
               st.write(choice)

        # Display answer
        answer = row['answer']
        st.write(f"Answer: {answer}")

        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    main()

