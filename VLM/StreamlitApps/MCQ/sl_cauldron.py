import time
from io import BytesIO

import streamlit as st
from datasets import load_dataset
from vlm_chat import LlavaChat

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(subject):
    try:
       ds = load_dataset("HuggingFaceM4/the_cauldron", subject)
       return ds['train']
    except Exception as e:
       st.error(f"Error loading dataset: {str(e)}")
       return None
    
def config_panel():
    st.sidebar.title("Cauldron")

    # Initialize the dictionary to hold the return values
    config = {
        "dataset": None,
        "start_index": 0,
        "end_index": 0
    }

    # Load dataset
    subject_options = ['ai2d', 'aokvqa', 'chart2text', 'chartqa', 'clevr', 'clevr_math', 'cocoqa', 'datikz', 
                       'diagram_image_to_text', 'docvqa', 'dvqa', 'figureqa', 'finqa', 'geomverse', 
                       'hateful_memes', 'hitab', 'iam', 'iconqa', 'infographic_vqa', 'intergps', 
                       'localized_narratives', 'mapqa', 'mimic_cgd', 'multihiertt', 'nlvr2', 'ocrvqa', 
                       'okvqa', 'plotqa', 'raven', 'rendered_text', 'robut_sqa', 'robut_wikisql', 
                       'robut_wtq', 'scienceqa', 'screen2words', 'spot_the_diff', 'st_vqa', 'tabmwp', 
                       'tallyqa', 'tat_qa', 'textcaps', 'textvqa', 'tqa', 'vistext', 'visual7w', 
                       'visualmrc', 'vqarad', 'vqav2', 'vsr', 'websight']
    
    subject = st.sidebar.selectbox("Select Subject", subject_options)  # Dropdown for subject selection
    dataset = load_data(subject)  # Use selected subject

    if dataset:
        config["dataset"] = dataset  # Store the loaded dataset
        num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
        total_items = len(dataset)
        total_pages = (total_items + num_items_per_page - 1) // num_items_per_page

        selected_page = st.sidebar.selectbox("Select Page Number", options=list(range(1, total_pages + 1)))

        # Calculate start and end index
        start_index = (selected_page - 1) * num_items_per_page
        end_index = min(start_index + num_items_per_page, total_items)

        # Update the config dictionary with actual values
        config["start_index"] = start_index
        config["end_index"] = end_index

    return config  # Return the config dictionary

@st.cache_resource
def load_vlm_model():
    vlm = LlavaChat()
    return vlm

def build_prompt(question, options=None):
    if not options:  # Check if options are empty
        prompt = f"You are an expert in mathematics. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are an expert in mathematics. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
    return prompt

def ask_vlm(question, options, image, index):
    if st.button(f"Ask VLM #{index}"):
        vlm = load_vlm_model()
        prompt = build_prompt(question, options)

        st.session_state.processing = True  # Set processing state
        with st.spinner("Retrieving answer..."):
            start_time = time.time()  # Start the timer
            try:
                answer = vlm.get_answer(prompt, image)  # Pass the byte stream
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                st.write(f"Model answer: {answer}")
                st.write(f"Elapsed time: {elapsed_time:.3f} seconds")  # Display elapsed time
            except Exception as e:
                st.error(f"Error retrieving answer: {str(e)}")
            finally:
                st.session_state.processing = False  # Reset processing state

def extract_data(data):
    questions = []
    options_list = []
    answers = []

    for item in data:
        user_text = item["user"]
        assistant_text = item["assistant"]
        
        # Extract question
        question_start = user_text.find("Question: ") + len("Question: ")
        question_end = user_text.find("\nChoices:")
        question = user_text[question_start:question_end].strip()
        questions.append(question)  # Append question to the list

        # Extract options
        options_start = user_text.find("\nChoices:\n") + len("\nChoices:\n")
        options_text = user_text[options_start:].split('\nAnswer with the letter.')[0]
        options = options_text.split("\n")
        options_list.append(options)  # Append options to the list

        # Extract answer
        answer_parts = assistant_text.split("Answer: ")
        if len(answer_parts) > 1:  # Check if the split was successful
            answer = answer_parts[1].strip()
        else:
            answer = "No answer provided."  # Handle the case where the answer is missing
        answers.append(answer)  # Append answer to the list

    return questions, options_list, answers

def process_question(row, qindex):
    st.header(f"Question: {qindex}")

    texts = row['texts']
    
    # Check for images in the row
    images = row.get('images', [])
    if images:
        for img in images:
            st.image(img)

    questions, options, answers = extract_data(texts)  # Update to unpack lists
    for question in questions:
        st.write("**Question**", question)

    for option in options:
        st.write("**Options**", option)

    for answer in answers:
        st.write("**Answer**", answer)

    st.divider()

def process_dataset():

    config = config_panel() 

    dataset = config["dataset"]
    if dataset is None:
        st.error("Dataset could not be loaded. Please try again.")
        return

    start_index = config["start_index"]
    end_index = config["end_index"]
    for i in range(start_index, end_index):
        process_question(dataset[i], i+1)

if __name__ == "__main__":
    process_dataset()
