import time
import string
from io import BytesIO

import streamlit as st
from PIL import Image
from datasets import load_dataset
from vlm_chat import LlavaChat

st.set_page_config(layout="wide")

# Define subsets
SUBSETS = [
    'Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 'Relative_Reflectance',
    'Visual_Correspondence', 'Counting', 'IQ_Test', 'Object_Localization',
    'Semantic_Correspondence', 'Visual_Similarity', 'Forensic_Detection', 'Jigsaw',
    'Relative_Depth', 'Spatial_Relation'
]

Experts = {
    "Art Style": ["Art Historians", "Aestheticians", "Visual Artists"],
    "Functional Correspondence": ["Cognitive Scientists", "Visual Cognition Experts", "AI/ML Researchers in Vision"],
    "Multi-view Reasoning": ["Spatial Cognition Experts", "3D Modeling Specialists", "Computer Vision Researchers"],
    "Relative Reflectance": ["Physicists (Optics)", "Visual Perception Experts", "Material Scientists"],
    "Visual Correspondence": ["Computer Vision Researchers", "Pattern Recognition Experts", "AI/ML Specialists"],
    "Counting": ["Cognitive Psychologists", "Numerical Cognition Experts", "Educational Psychologists"],
    "IQ Test": ["Psychometricians", "Cognitive Scientists", "Educational Testing Experts"],
    "Object Localization": ["Computer Vision Experts", "Robotics Researchers", "AI/ML Researchers"],
    "Semantic Correspondence": ["Linguists", "AI/ML Researchers", "Cognitive Scientists"],
    "Visual Similarity": ["Cognitive Psychologists", "Computer Vision Experts", "Pattern Recognition Specialists"],
    "Forensic Detection": ["Forensic Scientists", "Image Analysis Experts", "Criminal Investigators"],
    "Jigsaw": ["Puzzle Designers", "Cognitive Psychologists", "Spatial Reasoning Experts"],
    "Relative Depth": ["3D Graphics Experts", "Cognitive Psychologists", "Perception Scientists"],
    "Spatial Relation": ["Cognitive Scientists", "AI/ML Researchers", "Spatial Reasoning Experts"]
}


# Define splits
SPLITS = ["val", "test"]

# Load the dataset
@st.cache_data()
def load_data(subset, split):
    model_name = "BLINK-Benchmark/BLINK"

    try:
        ds = load_dataset(model_name, subset)
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

@st.cache_resource
def load_vlm_model():
    vlm = LlavaChat()
    return vlm

def build_prompt(question, options=None):
    if not options:  # Check if options are empty
        prompt = f"You are a helpful assistant. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are a helpful assistant. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
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

def process_question(row, index):
    st.header(f"Question #{index}")

    # Display question
    question = row['question']
    st.write(question)

    # Display images
    images = []
    for j in range(1, 5):
        image_key = f'image_{j}'
        if row[image_key] is not None:
            image = row[image_key]
            if image:
                images.append(image)
                st.image(image, caption=f"Image {j}", width=500)

    options = None
    if row['choices'] is not None:
        options = row['choices']
        for idx, option in enumerate(options):
            st.write(f"({chr(65 + idx)}) {option}")  # chr(65) is 'A'
    
    if st.button(f"Show Correct Answer #{index}"):
        st.write(row['answer'])

    ask_vlm(question, options, images, index)

    st.divider()


def config_panel():

    st.sidebar.title("BLINK")

    # Subset selection
    subset = st.sidebar.selectbox("Select Subset", SUBSETS)

    # Split selection
    split = st.sidebar.selectbox("Select Split", SPLITS)

    # Load dataset
    dataset = load_data(subset, split)

    if dataset:
        num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
        # Calculate total pages
        total_items = len(dataset)
        total_pages = (total_items // num_items_per_page) + (1 if total_items % num_items_per_page != 0 else 0)

        selected_page = st.sidebar.selectbox("Select Page Number", options=range(1, total_pages + 1))

    # Calculate start and end index
        start_index = (selected_page - 1) * num_items_per_page
        end_index = min(start_index + num_items_per_page, len(dataset))

        return dataset, start_index, end_index
    
    return None, 0, 0 

def process_dataset():

    dataset, start_index, end_index = config_panel() 

    if dataset is None: 
        st.error("Dataset could not be loaded. Please try again.")
        return

    for i in range(start_index, end_index):
        process_question(dataset[i], i+1)

# Streamlit app
if __name__ == "__main__":
    process_dataset()
