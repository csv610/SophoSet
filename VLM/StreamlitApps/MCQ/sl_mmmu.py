import streamlit as st
from datasets import load_dataset
import re

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(subset, split):
    """
    Load the specified subset and split of the AI2 ARC dataset.

    Args:
        subset (str): The subset of the dataset to load.
        split (str): The split of the dataset to load ('train', 'validation', or 'test').

    Returns:
        Dataset: The specified subset and split of the dataset.

    Raises:
        ValueError: If the dataset loading fails.
    """
    try:
        ds = load_dataset("MMMU/MMMU", subset)
        if split not in ds:
            raise ValueError(f"Invalid split '{split}' for subset '{subset}'")
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        raise ValueError(f"Failed to load dataset for subset '{subset}' and split '{split}'")

def config_panel():
    """
    Configure the sidebar panel for user interaction, including subset, split, and pagination settings.

    Returns:
        tuple: Contains the loaded dataset, number of items per page, and the selected page number.
    """
    st.sidebar.title("Navigation")

    # Subset selection
    subset = st.sidebar.selectbox("Select Subset", [ "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory", "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science", "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics", "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage", "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology" ])

    # Split selection
    split = st.sidebar.selectbox("Select Split", ["test", "validation", "dev"])

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

def process_latex(text):
    """
    Processes a string or a list of strings to escape dollar signs used for currency
    while preserving dollar signs used for LaTeX expressions.

    Args:
        text (str or list): The input string or list of strings containing both LaTeX expressions
                            and currency values.

    Returns:
        str or list: The processed string or list of strings with appropriate escaping.
    """
    if isinstance(text, list):
        # If the input is a list, process each item individually
        return [process_text(item) for item in text]

    # Identify LaTeX expressions in the text using regular expression
    latex_parts = re.findall(r'\$[^$]+\$', text)

    # Temporarily replace LaTeX parts with placeholders
    for i, part in enumerate(latex_parts):
        placeholder = f"__LATEX_{i}__"
        text = text.replace(part, placeholder)

    # Escape all remaining dollar signs (these are the non-LaTeX parts)
    text = text.replace('$', r'\$')

    # Restore the LaTeX expressions by replacing placeholders with original parts
    for i, part in enumerate(latex_parts):
        text = text.replace(f"__LATEX_{i}__", part)

    return text


# Streamlit app
def view_dataset():
    st.title("Dataset: MMMU")
    st.divider()

    # Get sidebar selections
    dataset, num_items_per_page, selected_page = config_panel()

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, len(dataset))

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {i + 1}")

        # Display question 
        question  = row['question']
        st.write(question)
        st.write("")  # Add vertical space

        for j in range(1,8):
            image_key = f"image_{j}"
            if row[image_key] is not None:
               image = row[image_key]
               st.image(image, caption = f"Qs: {i} Image{j}")
        st.write("")  # Add vertical space

         # Display  choices
        choices = row['options']
        st.write(choices)
        st.write("")  # Add vertical space

        if st.button(f"Human Answer: {i + 1}"):
            st.write(row['answer'])
            st.write("")  # Add vertical space

        st.divider()

if __name__ == "__main__":
    view_dataset()
