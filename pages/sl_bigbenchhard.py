import streamlit as st
from datasets import load_dataset
from PIL import Image
import os

st.set_page_config(layout="wide")

# Load the dataset with a specific subset (task)
@st.cache_data()
def load_data(subset):
    try:
       ds = load_dataset("maveriq/bigbenchhard", subset, trust_remote_code=True)
       return ds
    except Exception as e:
       st.error(f"Error loading dataset: {str(e)}")
       return None

# Function to load image from a local path
def load_image_from_path(image_path):
    return Image.open(image_path)

def split_text(text):
    """
    Splits the text into the main sentence and a clean list of options, then formats the options
    to include labels (e.g., (A), (B)) when returning them.

    Args:
        text (str): The full text containing the main sentence and options.

    Returns:
        tuple: A tuple containing the main sentence and the formatted options as a list.
    """
    # Find the last occurrence of the word "Options:"
    options_index = text.rfind("Options:")
    
    if options_index != -1:
        # Split the text into two parts
        main_sentence = text[:options_index].strip()
        options = text[options_index + len("Options:"):].strip()
        
        # Extract a clean list of options without labels
        clean_options = [opt.strip() for opt in options.split(")") if opt.strip()]
        
        # Format options with labels (A), (B), etc.
        formatted_options = [f"({chr(65 + i)}) {opt}" for i, opt in enumerate(clean_options)]
        
        return main_sentence, formatted_options
    else:
        return text, None

# Streamlit app
def main():
    st.title("BigBenchHard (BBH) Dataset")

    # List of subsets (tasks)
    subsets = [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting"
    ]

    # Dropdown to select the subset
    selected_subset = st.sidebar.selectbox("Select Subset", subsets)

    # Load dataset based on the selected subset
    dataset = load_data(selected_subset)

    split = "train"  # Assuming you're using the "train" split
    dataset = dataset[split]

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
        st.header(f"Question: {start_index + i + 1}")

        # Display input and target
        input  = row['input']
        target = row['target']

        st.write(f"{input}")
        st.write(f"Answer: {target}")

        st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

