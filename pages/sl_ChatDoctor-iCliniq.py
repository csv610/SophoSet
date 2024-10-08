import streamlit as st
from datasets import load_dataset
from llm_chat import LLMChat

st.set_page_config(layout="wide")

@st.cache_data()
def load_data(split='train'):
    try:
        ds = load_dataset("lavita/ChatDoctor-iCliniq")
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

@st.cache_resource
def load_llm_model(model_name: str = "llama3.1"):
    """Initialize and cache the LLM."""
    return LLMChat(model_name)

# Streamlit app
def main():
    st.title("Dataset: ChatDoctor iCliniq")
    st.divider()

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Load dataset
    dataset = load_data()
    if dataset is None:
        st.error("Failed to load the dataset. Please try again later.")
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

    llm = load_llm_model()

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {i + 1}")

        # Display question and answer
        question = row['input']
        answer   = row['answer_icliniq']

        st.write(question)
        st.write(f"Answer: {answer}")

        user_question = st.text_area(f"Edit Question {i+1}", height=100)
        if st.button(f"Ask Question:{i+1}"):
           with st.spinner("Processing ..."):
                prompt = "Based on the Context of " + question + " Answer the User Question: " + user_question
                response = llm.get_answer(prompt)
                st.write(f"Response: {response['response']}")
                st.write(f"Number of Input words: {response['num_input_words']}")
                st.write(f"Number of output  words: {response['num_output_words']}")
                st.write(f"Time: {response['response_time']}")

        st.divider()

if __name__ == "__main__":
    main()

