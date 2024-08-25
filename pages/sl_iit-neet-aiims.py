import streamlit as st
import pandas as pd
from llm_chat import LLMChat
import re

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data():
    """
    Load the CSV file into a pandas DataFrame.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        csv_file = "iit-neet-aiims.csv"  # Replace with your actual CSV file path
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

@st.cache_resource
def get_llm(model_name: str):
    """Initialize and cache the LLM."""
    return LLMChat(model_name)

def split_text(question_text):
    # Updated regex pattern to capture options more robustly
    options_pattern = r"([A-Da-d])\s*\.\s*(.*?)(?=\s*[A-Da-d]\s*\.\s*|$)"
    matches = re.findall(options_pattern, question_text)

    # Extract the question part before the first option
    question_part = re.split(options_pattern, question_text)[0].strip()

    # Build the options list, handling potential LaTeX or other special cases
    options = [match[1].strip() for match in matches]

    return question_part, options


def config_panel(df):
    """
    Configure the sidebar panel for user interaction, including pagination settings.

    Returns:
        tuple: Contains the DataFrame, number of items per page, and the selected page number.
    """
    st.sidebar.title("Navigation")

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=1)
    
    # Calculate total pages
    total_items = len(df)
    total_pages = (total_items + num_items_per_page - 1) // num_items_per_page

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    return df, num_items_per_page, selected_page

# Streamlit app
def view_dataset():
    """
    Display the dataset in a paginated format with questions and their subjects.
    """
    st.title("IIT-NEET-AIIMS Dataset")

    # Load data from CSV
    df = load_data()

    llm = get_llm("llama3.1")

    if df is not None:
        # Get sidebar selections
        dataset, num_items_per_page, selected_page = config_panel(df)

        # Display items for the selected page
        start_index = (selected_page - 1) * num_items_per_page
        end_index = min(start_index + num_items_per_page, len(dataset))

        for i in range(start_index, end_index):
            row = dataset.iloc[i]
            st.header(f"Question {start_index + i + 1}:")
            
            # Display question and subject
            question = row['eng']
            subject  = row['Subject']
            st.write(f"**Original:** {question}")

            question, options = split_text(question)
            st.write(f"**Question:** {question}")

            if options:
               st.write(f"**Options:** {options}")

            st.write(f"**Subject:** {subject}")
            st.write("")  # Add vertical space

            st.markdown("---")  # Divider between items

if __name__ == "__main__":
    view_dataset()

