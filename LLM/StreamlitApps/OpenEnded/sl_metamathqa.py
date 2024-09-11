import streamlit as st
from datasets import load_dataset
from llm_chat import LLMChat

# Load the dataset
@st.cache_data()
def load_data(split='train'):
    try:
        ds = load_dataset("meta-math/MetaMathQA")
        if split not in ds:
            raise ValueError(f"Invalid split: {split}. Available splits are: {', '.join(ds.keys())}")
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None
    
@st.cache_resource
def load_llm_model(model_name: str):
    """Initialize and cache the LLM."""
    return LLMChat(model_name)

def talk_to_llm(i, row, model_name):
    question = row['query']

    llm = load_llm_model(model_name)  # Changed from upload_llm_model to load_llm_model

    if st.button(f"LLM Answer:{i+1}"):
        with st.spinner("Processing ..."):

            response = llm.get_answer(question)
            st.write(f"Answer: {response['answer']}")
            st.write(f"Number of Input words: {response['num_input_words']}")
            st.write(f"Number of output  words: {response['num_output_words']}")
            st.write(f"Time: {response['response_time']}")

    if st.button(f"Ask Question: {i+1}"):
        user_question = st.text_area(f"Edit Question {i+1}", height=100)
        if user_question is not None:
              print( "USER QUESTION; ", user_question)
              with st.spinner("Processing ..."):
                   prompt = "Based on the Context of " + question + " Answer the User Question: " + user_question
                   response = llm.get_answer(prompt)
                   st.write(f"Answer: {response['answer']}")
                   st.write(f"Number of Input words: {response['num_input_words']}")
                   st.write(f"Number of output  words: {response['num_output_words']}")
                   st.write(f"Time: {response['response_time']}")

# Streamlit app
def main():
    st.title("Dataset: MetaMathQA")
    st.divider()

    # Load dataset
    dataset = load_data()
    if dataset is None:
        st.stop()

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

    model_name = "llama3.1"
    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question {i + 1}")
        st.write(row['query'])

        if st.button(f"Human Answer :{i+1}"):
           st.write(f"Answer: {row['response']}")

        talk_to_llm(i+1, row, model_name)

        st.divider()  # Divider between items

if __name__ == "__main__":
    main()
