import time  # Ensure
import os

from PIL import Image
import streamlit as st
from datasets import load_dataset
from vlm_chat import LlavaModel

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    model_name = "Zhiqiang007/MathV360K"
    try:
        dataset = load_dataset(model_name)
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found in the dataset.")
        return dataset[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Function to display conversation with hint, question, and choices on separate lines
def display_conversation(conversation_list):
    for entry in conversation_list:
        speaker = entry["from"]
        message = entry["value"]
        
        if speaker == "human":
            lines = message.split("\n")
            for line in lines:
                if line.startswith("<image>") or line.startswith("Hint:"):
                    continue
                st.write(line)
        
        elif speaker == "gpt":
            st.write(f"**GPT:** {message}")

@st.cache_resource
def load_vlm_model():
    vlm = LlavaModel()
    return vlm

def build_prompt(question, options=None):
    if not options:  # Check if options are empty
        prompt = f"You are an expert in mathematics. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are an expert in mathematics. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
    return prompt

def ask_vlm(params):
    question = params["question"]
    options = params["options"]
    image = params["image"]
    index = params["index"]
    # Use a button to trigger the answer retrieval
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

def extract_data(conversation_list):
    question = None
    answer = None
    options = None 
    
    for entry in conversation_list:
        if entry["from"] == "human":
            # Extract question and options from the human's message
            if "Question:" in entry["value"]:
                question_part = entry["value"].split("Question:")[-1].strip()
                question, options_part = question_part.split("Choices:") if "Choices:" in question_part else (question_part, None)
                if options_part:
                    options = [opt.strip() for opt in options_part.split("\n") if opt.strip()]  # Parse options
            
        elif entry["from"] == "gpt":
            # Extract answer from the GPT's message
            answer = entry["value"].strip()
    
    return question, options, answer  # Return options along with question and answer

def process_question(row, index, image_folder):
    st.header(f"Question: {index + 1}")

    question, options, answer = extract_data(row['conversations'])

    st.write(question)

    # Check if the image folder exists
    if not os.path.exists(image_folder):
        st.error(f"Image folder does not exist. Download images from Huggingface and place in the local folder {image_folder}.")
        return  # Exit the function if the folder does not exist

    image = os.path.join(image_folder, row['image'])

    if options:
        for opt in options:
            st.write(opt)
    
    # Check if the image was loaded successfully
    if image is not None:
        st.image(image, caption=f"Qs: {index}", use_column_width=True)
    
    if st.button(f"Show Correct Answer #{index}"):  # Updated button label
        if answer:  # Check if answer is not None
            st.write(answer)
        else:
            st.write("No answer available.")  # Inform the user if no answer is found

    # Create a dictionary to hold the parameters
    vlm_params = {
        "question": question,
        "options": None,  # Assuming options is None
        "image": image,
        "index": index
    }
    
    ask_vlm(vlm_params)  # Pass the dictionary to the function

def config_panel():
    st.sidebar.title("MathV360K")

    # Initialize return dictionary
    config = {
        "dataset": None,
        "start_index": 0,
        "end_index": 0,
        "image_folder": None
    }

    # Load dataset
    split = 'train'
    config["dataset"] = load_data(split)
    
    if config["dataset"] is None:  # Check if dataset is None
        st.error("Dataset could not be loaded. Please try again.")
        return config  # Return the initialized config

    # Ask for image folder
    config["image_folder"] = st.sidebar.text_input("Enter Image Folder Path", "../../images/mathv360k")
    
    # Select number of questions per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Calculate total pages
    total_items = len(config["dataset"])
    total_pages = (total_items // num_items_per_page) + 1

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    # Calculate start and end index for pagination
    config["start_index"] = (selected_page - 1) * num_items_per_page
    config["end_index"] = min(config["start_index"] + num_items_per_page, total_items)

    return config

# Streamlit app
def process_dataset():
    config = config_panel()
    dataset = config["dataset"]
    
    if dataset is None:  # Check if dataset is None
        st.error("Dataset could not be loaded. Please try again.")
        return  # Exit the function if dataset is not loaded
    
    start_index = config["start_index"]
    end_index = config["end_index"]
    image_folder = config["image_folder"]

    # Display items for the selected page
    for i in range(start_index, end_index):
        process_question(dataset[i], i+1, image_folder)

if __name__ == "__main__":
    process_dataset()


