import streamlit as st
from datasets import load_dataset
from typing import List, Tuple, Dict, Optional
from llm_chat import LLMChat
import logging

# Constants
DATASET_NAME = "openlifescienceai/medmcqa"
SPLITS = ["train", "validation", "test"]
MODEL_OPTIONS = ["llama3.1", "llama2", "gpt-3.5-turbo"]
MAX_ITEMS_PER_PAGE = 10
ANSWER_OPTIONS = ['A', 'B', 'C', 'D']

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(layout="wide", page_title="Medmcqa Dataset")

@st.cache_data
def load_data(split: str) -> Dict:
    """
    Load and cache the dataset for a given split.
    
    Args:
        split (str): The dataset split to load.
    
    Returns:
        Dict: The loaded dataset.
    """
    try:
        ds = load_dataset(DATASET_NAME)
        return ds[split]
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        st.error(f"Failed to load dataset: {e}")
        return {}

def get_answer_options(row: Dict) -> List[str]:
    """
    Extract answer options from a dataset row.
    
    Args:
        row (Dict): A row from the dataset.
    
    Returns:
        List[str]: List of answer options.
    """
    return [row['opa'], row['opb'], row['opc'], row['opd']]

def display_question(row: Dict, index: int):
    """
    Display a question and its options.
    
    Args:
        row (Dict): A row from the dataset.
        index (int): The index of the question.
    """
    st.header(f"Question: {index + 1}")
    st.write(row['question'])
    for option, text in zip(ANSWER_OPTIONS, get_answer_options(row)):
        st.write(f"({option}) {text}")
    st.write(f"Answer: ({ANSWER_OPTIONS[int(row['cop'])]})")
    st.write(f"Explanation: {row['exp']}")

def get_model_answer(llm: LLMChat, question: str, options: List[str]) -> Tuple[str, str]:
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
        answer, explanation = llm.generate_answer(question, options)
        # Convert numeric answer to letter
        if answer.isdigit() and 1 <= int(answer) <= 4:
            answer = f"({ANSWER_OPTIONS[int(answer)-1]})"
        return answer, explanation
    except Exception as e:
        logger.error(f"Error generating model answer: {e}")
        return "Error", f"Failed to generate answer: {e}"

def get_model_explanation(llm: LLMChat, question: str) -> str:
    """
    Get an explanation from the model.
    
    Args:
        llm (LLMChat): The LLM chat instance.
        question (str): The question to explain.
    
    Returns:
        str: The model's explanation.
    """
    try:
        return llm.generate_explanation(question)
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return f"Failed to generate explanation: {e}"

def sidebar_navigation() -> Tuple[str, int, int, str]:
    """
    Create and handle sidebar navigation.
    
    Returns:
        Tuple[str, int, int, str]: Selected split, page, items per page, and model.
    """
    st.sidebar.title("Navigation")
    split = st.sidebar.selectbox("Select Split", SPLITS, index=0)
    num_items_per_page = st.sidebar.slider("Items per Page", 1, MAX_ITEMS_PER_PAGE, 5)
    
    dataset = load_data(split)
    total_items = len(dataset)
    if total_items == 0:
        st.sidebar.error("Dataset is empty. Please check your connection or try another split.")
        return split, 1, num_items_per_page, MODEL_OPTIONS[0]
    
    total_pages = (total_items - 1) // num_items_per_page + 1
    page = st.sidebar.selectbox("Select Page", range(1, total_pages + 1), index=0)
    model_name = st.sidebar.selectbox("Select Model", MODEL_OPTIONS, index=0)
    
    return split, page, num_items_per_page, model_name

def main():
    st.title("Medmcqa Dataset")

    split, page, items_per_page, model_name = sidebar_navigation()
    
    if not all([split, page, items_per_page, model_name]):
        st.error("Error in navigation settings. Please refresh the page and try again.")
        return

    dataset = load_data(split)
    if not dataset:
        st.error("Failed to load dataset. Please check your connection and try again.")
        return

    llm = LLMChat(model_name)

    start_index = (page - 1) * items_per_page
    end_index = min(start_index + items_per_page, len(dataset))

    for i in range(start_index, end_index):
        row = dataset[i]
        display_question(row, i)

        if st.button("Get Model Answer", key=f"answer_{i}"):
            with st.spinner('Getting model answer...'):
                answer, explanation = get_model_answer(llm, row['question'], get_answer_options(row))
                st.write(f"**Model's Answer:** {answer}")
                st.write(f"**Model's Explanation:** {explanation}")
                
                correct_answer = f"({ANSWER_OPTIONS[int(row['cop'])-1]})"
                if answer == correct_answer:
                    st.success("The model's answer matches the correct answer!")
                else:
                    st.warning(f"The model's answer differs from the correct answer {correct_answer}.")

        explain_question = st.text_input("Ask for additional explanation:", key=f"explain_input_{i}")
        if explain_question and st.button("Explain", key=f"explain_button_{i}"):
            with st.spinner('Getting explanation from model...'):
                explanation = get_model_explanation(llm, explain_question)
                st.write(f"**Additional Explanation:** {explanation}")

        st.markdown("---")

if __name__ == "__main__":
    main()
