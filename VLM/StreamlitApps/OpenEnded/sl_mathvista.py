import streamlit as st
from datasets import load_dataset

from vlm_chat import LlavaChat
from io import BytesIO
import time 

st.set_page_config(layout="wide")

def load_vlm_model():
    vlm = LlavaChat()
    return vlm

def build_prompt(question, options):
    if not options:  # Check if options are empty
        prompt = f"You are an expert in mathematics. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are an expert in mathematics. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
    return prompt

def ask_vlm(question, options, image, index):
    
    # Convert image to bytes if it exists
    if image:
        img_byte_arr = BytesIO()
        # Convert image to RGB mode if it is in RGBA
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(img_byte_arr, format='JPEG')  # Change 'JPG' to 'JPEG'
        img_byte_arr.seek(0)  # Move to the beginning of the byte stream
        image = img_byte_arr  # Update image to byte stream

    # Use a button to trigger the answer retrieval
    if st.button(f"Ask VLM : {index}"):
        st.session_state.processing = True  # Set processing state
        with st.spinner("Retrieving answer..."):
            start_time = time.time()  # Start the timer
            try:
                print(prompt)
                answer = vlm.get_answer(prompt, image)  
                elapsed_time = time.time() - start_time 
                st.write(f"Model answer: {answer}")
                st.write(f"Elapsed time: {elapsed_time:.3f} seconds")  # Display elapsed time
            except Exception as e:
                st.error(f"Error retrieving answer: {str(e)}")
            finally:
                st.session_state.processing = False  # Reset processing state

def explore_data():

    st.sidebar.title("MathVista")

    # Sidebar for all the settings
    # Select the dataset split
    opt_split = st.sidebar.selectbox("Select dataset split", ["test", "testmini"])

    # Load the dataset
    dataset = load_dataset("AI4Math/MathVista", split=opt_split)

    # Select the question type
    opt_question_type = st.sidebar.selectbox("Select question type", ["all", "multi_choice", "free_form"])

    # Filter the dataset based on the selected question type
    if opt_question_type != "all":
        dataset = dataset.filter(lambda x: x['question_type'] == opt_question_type)

    # Select the category
    opt_category = st.sidebar.selectbox("Select category", ["all", "math-targeted-vqa", "general-vqa"])

    # Filter the dataset based on the selected category
    if opt_category != "all":
        dataset = dataset.filter(lambda x: x['metadata']['category'] == opt_category)

    # Select the grade level
    opt_grade = st.sidebar.selectbox("Select grade level", ["all", "elementary school", "high school", "college", "not applicable"])

    # Filter the dataset based on the selected grade level
    if opt_grade != "all":
        dataset = dataset.filter(lambda x: x['metadata']['grade'].lower() == opt_grade.lower())

    # Select the context
    opt_context = st.sidebar.selectbox("Select context", ["all", "abstract scene", "bar chart", "document image", 
    "function plot", "geometry diagram", "heatmap chart", "line plot", "map chart", "medical image", 
    "natural image", "pie chart", "puzzle test", "radar chart", "scatter plot", "scientific image", 
    "synthetic scene", "table", "violin plot", "word cloud"])

    # Filter the dataset based on the selected context
    if opt_context != "all":
        dataset = dataset.filter(lambda x: x['metadata']['context'].lower() == opt_context.lower())

    # Select the skill
    opt_skill = st.sidebar.selectbox("Select Skill", ["all", "algebraic reasoning", "arithmetic reasoning",  
    "geometry reasoning", "logical reasoning", "numeric commonsense", "scientific reasoning", 
    "statistical reasoning"])

    # Filter the dataset based on the selected skill
    if opt_skill != "all":
        dataset = dataset.filter(lambda x: opt_skill.lower() in [skill.lower() for skill in x['metadata']['skills']])

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

    vlm = load_vlm_model()

    for i in range(start_index, end_index):
        row = dataset[i]

        st.header(f"Question: {i+1}")
        question = row['question']
        st.write(question)

        image = None
        if row['decoded_image']:
            image = row['decoded_image']
            st.image(image, caption=f"Qs:{i+1}")
                
        options = None
        if row['choices']:
            options = [f"({chr(65 + i)}) {choice}" for i, choice in enumerate(row['choices'])]
            for choice in options:
                st.write(choice)

        if row['answer']:
            if st.button(f"Human Answer:{i+1}"):
                st.write(row['answer'])

        ask_vlm(question, options, image, i+1)

        st.divider()

if __name__ == "__main__":
    explore_data()

