import streamlit as st
from datasets import load_dataset
from llm_chat import load_llm_model, ask_llm

st.set_page_config(layout="wide")

# Constants
MODEL_ID = "GBaker/MedQA-USMLE-4-options-hf"
SPLITS = ["train", "validation", "test"]

# Load the dataset
@st.cache_data()
def load_data(split):
    try:
       ds = load_dataset(MODEL_ID)
       return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None
    
def process_question(llm, row, index):
    st.header(f"Question #{index}")

        # Display question and answer
    question = row['sent1']
    st.write(question)

        # Create a list of options using row items
    options = [row[f'ending{i}'] for i in range(4)]  # Assuming there are 4 options: ending0 to ending3
        
        # Display options with labels
    for i, option in enumerate(options):
        st.write(f"({chr(65 + i)}) {option}")  # chr(65) is 'A'

    # Displaying answer with a button
    if st.button(f"Correct Answer#{index}"):
        st.write(f"Answer: {chr(65+row['label'])}")

    ask_llm(llm, question, options, index)

# Streamlit app
def main():
    st.sidebar.title("MedQA-USMLE-4")

    # Split selection
    split = st.sidebar.selectbox("Select split", SPLITS)

    # Load dataset
    dataset = load_data(split)

    if dataset is None:
        st.error("No data available for the selected split.")
        return

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

    # Select model using Streamlit selectbox
    model_name = st.selectbox("Select model:", ["llama3.1", "llama3.2"], index=0)  # Default to llama3.1
    llm = load_llm_model(model_name)

    for i in range(start_index, end_index):
        process_question(llm, dataset[i], i+1)

    st.divider()

if __name__ == "__main__":
    main()

