import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import re
from llm_chat import LLMChat
from typing import List, Tuple, Dict, Optional
import logging

st.set_page_config(layout="wide")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the dataset
@st.cache_data()
def load_data(subset, split):
    ds = load_dataset("lighteval/MATH", subset)
    return ds[split]

def get_mcq_answer(llm: LLMChat, question: str, options: List[str]) -> Tuple[str, str]:
    """
    Get an answer from the model.

    Args:
        llm (LLMChat): The LLM chat instance.
        question (str): The question to answer.
        options (List[str]): The answer options.

    Returns:
        Tuple[str, str]: The model's answer and explanation.
    """
    try:
        answer, explanation = llm.get_mcq_answer(question, options)
        # Convert numeric answer to letter
        if answer.isdigit() and 1 <= int(answer) <= 4:
            answer = f"({ANSWER_OPTIONS[int(answer)-1]})"
        return answer, explanation
    except Exception as e:
        logger.error(f"Error generating model answer: {e}")
        return "Error", f"Failed to generate answer: {e}"


def get_answer(llm: LLMChat, question: str) -> str:
    """
    Get an explanation from the model.

    Args:
        llm (LLMChat): The LLM chat instance.
        question (str): The question to explain.

    Returns:
        str: The model's explanation.
    """
    try:
        return llm.get_answer(question)
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return f"Failed to generate explanation: {e}"

# Streamlit app
def view_dataset():
    st.title("MATH Dataset")

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Subset selection
    subset = st.sidebar.selectbox("Select Subset", [
        "all",
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus"
    ])

    # Split selection
    split = st.sidebar.selectbox("Select Split", ["train", "test"])

    # Load dataset
    dataset = load_data(subset, split)

    if not dataset:
        st.error("Failed to load dataset. Please check your connection and try again.")
        return


    # Model selection
    model_name = st.sidebar.selectbox("Select Model", ["llama3.1", "llama2", "gpt-3.5-turbo"], key="model_selector")

    llm = LLMChat(model_name)

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

        # Display question and answer
        problem = row['problem']
        solution = row['solution']

        st.write(problem)
        st.write("")
        st.write(f"Solution: {solution}")

        explain_question = st.text_input("Ask for additional explanation:", key=f"explain_input_{i}")
        if explain_question and st.button("Explain", key=f"explain_button_{i}"):
            with st.spinner('Getting explanation from model...'):
                explanation = get_model_explanation(llm, explain_question)
                st.write(f"**Additional Explanation:** {explanation}")


        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    view_dataset()

