import streamlit as st
from datasets import load_dataset
from llm_chat import load_llm_model, ask_llm

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    try:
        ds = load_dataset("openlifescienceai/medmcqa")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return []  # Return an empty list or handle as needed
    return ds[split]

def process_question(row, index):
    st.header(f"Question: {index}")

    # Display question and answer
    question = row['question']
    opa = row['opa']
    opb = row['opb']
    opc = row['opc']
    opd = row['opd']
    cop = row['cop']
    exp = row['exp']

    st.write(question)
    st.write(f"(A): {opa}")
    st.write(f"(B): {opb}")
    st.write(f"(C): {opc}")
    st.write(f"(D): {opd}")

    # Displaying answer with a button
    if st.button(f"Correct Answer #{index}"):
        st.write(f"Answer: {chr(65 + cop)}")

        # Displaying explanation with a button
    if st.button(f"Explanation:{index}"):
        st.write(f"Explanation #{exp}")

    st.divider()

# Streamlit app
def process_dataset():
    st.sidebar.title("Medmcqa")

    # Split selection
    split = st.sidebar.selectbox("Select Split", ["train", "validation", "test"])

    # Load dataset
    dataset = load_data(split)

    # Check if the dataset is empty
    if not dataset:
        st.warning("No data available for the selected split.")
        return  # Exit the function if the dataset is empty

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

    model_name = st.sidebar.selectbox("Select Model", ["llama3.1", "llama3.2"])
    llm = load_llm_model(model_name)

    for i in range(start_index, end_index):
        process_question(dataset[i], i+1)

if __name__ == "__main__":
    process_dataset()

