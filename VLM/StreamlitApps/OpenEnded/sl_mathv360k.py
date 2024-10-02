import time  # Ensure
import os

from PIL import Image
import streamlit as st
from datasets import load_dataset
from vlm_chat import LlavaChat

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

# Function to load image from a file
def load_image_from_file(image_filename, image_folder="./images/mathv360k_images"):
    image_path = os.path.join(image_folder, image_filename)
    
    # Check if the image path exists
    if not os.path.exists(image_path):
        st.error(f"Image not found: {image_path}")
        return None  # Return None if the image does not exist
    
    return Image.open(image_path)

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

@st.cache_resource()  # Add caching to the model loading function
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

def extract_question_answer(conversation_list):
    question = None
    answer = None
    
    for entry in conversation_list:
        if entry["from"] == "human":
            # Extract question from the human's message
            if "Question:" in entry["value"]:
                question = entry["value"].split("Question:")[-1].strip()
        
        elif entry["from"] == "gpt":
            # Extract answer from the GPT's message
            answer = entry["value"].strip()
    
    return question, answer

def process_question(row, index, image_folder):
    st.header(f"Question: {index + 1}")

    question, answer = extract_question_answer(row['conversations'])

    st.write(question)

    # Display image (from file)
    image = load_image_from_file(row['image'], image_folder)  # Pass the user-defined image folder
    
    # Check if the image was loaded successfully
    if image is not None:
        st.image(image, caption=f"Qs: {index}", use_column_width=True)
    
    if st.button(f"Show Correct Answer #{index}"):  # Updated button label
        if answer:  # Check if answer is not None
            st.write(answer)
        else:
            st.write("No answer available.")  # Inform the user if no answer is found

    ask_vlm(question, None, image, index)

def config_panel():
    # Load dataset
    split = 'train'
    dataset = load_data(split)

    if dataset is None:
        st.stop()  # Stop execution if dataset loading failed

    # Sidebar for navigation
    st.sidebar.title("MathV360K")

    # Ask for image folder
    image_folder = st.sidebar.text_input("Enter Image Folder Path", "../../images/mathv360k")
    
    # Select number of questions per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + 1

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    # Calculate start and end index for pagination
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, total_items)

    return dataset, start_index, end_index, image_folder

# Streamlit app
def process_dataset():
    dataset, start_index, end_index, image_folder = config_panel()
    
    if dataset is None:  # Check if dataset is None
        st.error("Dataset could not be loaded. Please try again.")
        return  # Exit the function if dataset is not loaded

    # Display items for the selected page
    for i in range(start_index, end_index):
        process_question(dataset[i], i+1, image_folder)

if __name__ == "__main__":
    process_dataset()


