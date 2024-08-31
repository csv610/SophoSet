import streamlit as st
from datasets import load_dataset
import re

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(subset, split):
    model_name  = "allenai/ai2_arc"
    try:
       ds = load_dataset(model_name, subset)
       return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def rewrite_sentence(text):
    """
    Rewrite a sentence to escape LaTeX dollar signs and ensure money signs are not escaped.

    Args:
        text (str): The text to process.

    Returns:
        str: The processed text with LaTeX expressions escaped and money signs handled correctly.
    """
    # First, identify and escape LaTeX dollar signs
    text = re.sub(r'(?<!\\)\$(.*?)(?<!\\)\$', r'\\$\1\\$', text)

    # Then, identify and ensure money dollar signs are not escaped
    text = re.sub(r'\\\$(\d+(\.\d{1,2})?)\\\$', r'$\1$', text)

    return text

def config_panel():
    """
    Configure the sidebar panel for user interaction, including subset, split, and pagination settings.

    Returns:
        tuple: Contains the loaded dataset, number of items per page, and the selected page number.
    """
    st.sidebar.title("Navigation")

    # Subset selection
    subset = st.sidebar.selectbox("Select Subset", ["ARC-Challenge", "ARC-Easy"])

    # Split selection
    split = st.sidebar.selectbox("Select Split", ["train", "validation", "test"])

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Load dataset
    dataset = load_data(subset, split)

    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items + num_items_per_page - 1) // num_items_per_page

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    return dataset, num_items_per_page, selected_page

# Streamlit app
def view_dataset():
    """
    Display the AI2 ARC dataset in a paginated format with questions and choices labeled as (A), (B), etc.
    """
    st.title("AI2 ARC Dataset")
    st.divider()  

    # Get sidebar selections
    dataset, num_items_per_page, selected_page = config_panel()

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, len(dataset))

    choice_labels = ["A", "B", "C", "D", "E"]
    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {i + 1}")

        # Display question and answer
        st.write(row['question'])
        st.write(" ")

        choices = row['choices']['text']
        # Format the choices as (A), (B), etc.
        formatted_choices = [f"({choice_labels[j]}) {choices[j]}" for j in range(len(choices))]

        for formatted_choice in formatted_choices:
            st.write(formatted_choice)
        st.write("")  # Add vertical space

        st.write(f"Answer: {row['answerKey']}")

        st.divider()  

if __name__ == "__main__":
    view_dataset()

