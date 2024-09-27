import streamlit as st
from datasets import load_dataset
from vlm_chat import LlavaChat
import time  # Ensure to import time at the top of the file

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    model_name = "MathLLMs/MathVision"
    try:
        ds = load_dataset(model_name)
        return ds[split]
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def load_vlm_model():
    vlm = LlavaChat()
    return vlm

def build_prompt(question, options):
    if not options:  # Check if options are empty
        prompt = f"You are an expert in mathematics. You are given an open-ended question: '{question}'. Provide a detailed answer."
    else:
        prompt = f"You are an expert in mathematics. You are given a question '{question}' with the following options: {options}. Think step by step before answering the question and select the best option that answers the question as correctly as possible."
    return prompt
    
# Streamlit app
def main():
    # Add a session state variable to track the processing state
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    print("Step 1 ")

    # Sidebar for navigation
    st.sidebar.title("MathVision")
    
    # Select subset
    split = st.sidebar.selectbox("Select Dataset Subset", options=['test', 'testmini'])

    # Load dataset
    dataset = load_data(split)
    print("Step 2 ")
    
    if dataset is None:
        st.warning("Unable to load dataset. Please try again later.")
        return

    # Select number of questions per page
    num_questions_per_page = st.sidebar.slider("Select Number of Questions per Page", min_value=1, max_value=10, value=5)
    
    # Calculate total pages
    total_questions = len(dataset)
    total_pages = (total_questions // num_questions_per_page) + 1

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

    # Display questions for the selected page
    start_index = (selected_page - 1) * num_questions_per_page
    end_index = min(start_index + num_questions_per_page, total_questions)

    vlm = load_vlm_model()

    # Cancel button to stop processing
    if st.session_state.processing and st.sidebar.button("Cancel"):
        st.session_state.processing = False  # Logic to cancel the process
        st.warning("Processing canceled.")

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question {i + 1}")

        question = row['question']
        st.write(question)
        print(question)

        image = None
        if row['decoded_image']:
            image = row['decoded_image']
            st.image(image, caption=f"Question {i + 1}", use_column_width=True)

        # Convert image to bytes if it exists
        if image:
            from io import BytesIO
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='JPEG')  # Save as JPEG
            img_byte_arr.seek(0)  # Move to the beginning of the byte stream
            image = img_byte_arr  # Update image to byte stream

        options = None
        if row['options'] and len(row['options']) > 0:
            options = row['options']
            st.write(options)

        if row['answer']:
            if st.button(f"Human Answer: {i + 1}"):
                st.write(row['answer'])

        prompt = build_prompt(question, options)
        
        # Use a button to trigger the answer retrieval
        if st.button(f"Ask VLM : {i + 1}"):
            st.session_state.processing = True  # Set processing state
            with st.spinner("Retrieving answer..."):
                start_time = time.time()  # Start the timer
                try:
                    print(prompt)
                    answer = vlm.get_answer(prompt, image)  # Pass the byte stream
                    elapsed_time = time.time() - start_time  # Calculate elapsed time
                    st.write(f"Model answer: {answer}")
                    st.write(f"Elapsed time: {elapsed_time:.3f} seconds")  # Display elapsed time
                except Exception as e:
                    st.error(f"Error retrieving answer: {str(e)}")
                finally:
                    st.session_state.processing = False  # Reset processing state

        st.divider()

if __name__ == "__main__":
    main()
