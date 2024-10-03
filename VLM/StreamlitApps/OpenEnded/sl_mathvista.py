import time 

import streamlit as st
from datasets import load_dataset
from vlm_chat import LlavaModel
from io import BytesIO

st.set_page_config(layout="wide")

# Constants
QUESTION_HEADER = "Question: "
HUMAN_ANSWER_BUTTON = "Human Answer:"
VLM_ASK_BUTTON = "Ask VLM : "

# Constants for context options
CONTEXT_OPTIONS = [
    "all", "abstract scene", "bar chart", "document image", 
    "function plot", "geometry diagram", "heatmap chart", "line plot", 
    "map chart", "medical image", "natural image", "pie chart", 
    "puzzle test", "radar chart", "scatter plot", "scientific image", 
    "synthetic scene", "table", "violin plot", "word cloud"
]
# Constants for skill options
SKILL_OPTIONS = [
        "all", "algebraic reasoning", "arithmetic reasoning",  
        "geometry reasoning", "logical reasoning", "numeric commonsense", 
        "scientific reasoning", "statistical reasoning"
]

@st.cache_resource
def load_vlm_model():
    vlm = LlavaModel()
    return vlm

def build_prompt(question, options):
    if not options:  # Check if options are empty
        prompt = f"You are an expert in mathematics. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are an expert in mathematics. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
    return prompt

def generate_options(choices):
    """Generate formatted options from choices."""
    return [f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)]

def ask_vlm(question, options, image, index):
    
    # Use a button to trigger the answer retrieval
    if st.button(f"Ask VLM : {index}"):
        vlm = load_vlm_model()
        prompt = build_prompt(question, options)

        st.session_state.processing = True  # Set processing state
        with st.spinner("Retrieving answer..."):
            start_time = time.time()  # Start the timer
            try:
                answer = vlm.get_answer(prompt, image)  
                elapsed_time = time.time() - start_time 
                st.write(f"Model answer: {answer}")
                st.write(f"Elapsed time: {elapsed_time:.3f} seconds")  # Display elapsed time
            except Exception as e:
                st.error(f"Error retrieving answer: {str(e)}")
            finally:
                st.session_state.processing = False  # Reset processing state

def process_question(row: dict, index: int):
    st.header(f"{QUESTION_HEADER}{index}")
    st.write(row['question'])

    image = row['decoded_image']
    if image:
        st.image(image, caption=f"Qs:{index}")

    options = generate_options(row['choices']) if row['choices'] else []
    for choice in options:
        st.write(choice)

    if row['answer'] and st.button(f"{HUMAN_ANSWER_BUTTON}{index}"):
        st.write(row['answer'])

    ask_vlm(row['question'], options, image, index)
    st.divider()

def config_panel():
    st.sidebar.title("MathVista")

    # Initialize the dictionary to hold the return values
    config = {
        "dataset": None,
        "start_index": 0,
        "end_index": 0
    }

    # Select the dataset split
    opt_split = st.sidebar.selectbox("Select dataset split", ["test", "testmini"])

    # Load the dataset
    dataset = load_dataset("AI4Math/MathVista", split=opt_split)
    config["dataset"] = dataset  # Store the loaded dataset

    # Select the question type
    opt_question_type = st.sidebar.selectbox("Select question type", ["all", "multi_choice", "free_form"])

    # Filter the dataset based on the selected question type
    if opt_question_type != "all":
        config["dataset"] = config["dataset"].filter(lambda x: x['question_type'] == opt_question_type)

    # Select the category
    opt_category = st.sidebar.selectbox("Select category", ["all", "math-targeted-vqa", "general-vqa"])

    # Filter the dataset based on the selected category
    if opt_category != "all":
        config["dataset"] = config["dataset"].filter(lambda x: x['metadata']['category'] == opt_category)

    # Select the grade level
    opt_grade = st.sidebar.selectbox("Select grade level", ["all", "elementary school", "high school", "college", "not applicable"])

    # Filter the dataset based on the selected grade level
    if opt_grade != "all":
        config["dataset"] = config["dataset"].filter(lambda x: x['metadata']['grade'].lower() == opt_grade.lower())

    # Update the selectbox to use the constant
    opt_context = st.sidebar.selectbox("Select context", CONTEXT_OPTIONS)

    # Filter the dataset based on the selected context
    if opt_context != "all":
        config["dataset"] = config["dataset"].filter(lambda x: x['metadata']['context'].lower() == opt_context.lower())

    # Update the selectbox to use the constant
    opt_skill = st.sidebar.selectbox("Select Skill", SKILL_OPTIONS)

    # Filter the dataset based on the selected skill
    if opt_skill != "all":
        config["dataset"] = config["dataset"].filter(lambda x: opt_skill.lower() in [skill.lower() for skill in x['metadata']['skills']])

    if config["dataset"] is not None:  # Check if the dataset is valid
        num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
        
        # Calculate total pages
        total_items = len(config["dataset"])
        total_pages = (total_items // num_items_per_page) + (1 if total_items % num_items_per_page != 0 else 0)

        # Page selection box
        page_options = list(range(1, total_pages + 1))
        selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

        # Calculate start and end index
        start_index = (selected_page - 1) * num_items_per_page
        end_index = min(start_index + num_items_per_page, total_items)

        # Update the config dictionary with actual values
        config["start_index"] = start_index
        config["end_index"] = end_index

    return config  # Return the config dictionary

def process_dataset():
    config = config_panel()  # Call config_panel and store the returned dictionary
    dataset = config["dataset"]
    start_index = config["start_index"]
    end_index = config["end_index"]
    
    if dataset is None:  # Check if dataset is None
        st.error("Dataset could not be loaded. Please try again.")
        return  # Exit the function if dataset is not loaded

    for i in range(start_index, end_index):
        process_question(dataset[i], i + 1)

if __name__ == "__main__":
    process_dataset()

