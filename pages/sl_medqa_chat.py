import streamlit as st
from datasets import load_dataset
from llm_chat import LLMChat

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    return load_dataset("openlifescienceai/medqa", split=split)

# Function to reset session state for the current index
def reset_session_state(index):
    st.session_state[f"model_answer_{index}"] = ""
    st.session_state[f"explain_input_{index}"] = ""

# Function to display question and options
def display_question_options(data, index):
    question = data.get('Question', 'No question available')
    options  = data.get('Options', {})
    answer   = data.get('Correct Option', 'N/A')

    st.header(f"Question: {index + 1}")
    st.write(question)
    for label, option in options.items():
        st.write(f"{label}: {option}")
    st.write(f"Answer: {answer}")

def main():
    st.title("MedQA Dataset")

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Split selection
    split = st.sidebar.selectbox("Select Split", ["train", "test", "dev"])

    # Load dataset
    dataset = load_data(split)

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items + num_items_per_page - 1) // num_items_per_page

    # Page selection box
    selected_page = st.sidebar.selectbox("Select Page Number", options=range(1, total_pages + 1))

    # Model selection
    model_name = st.sidebar.selectbox("Select Model", ["llama3.1", "llama2", "gpt-3.5-turbo"], key="model_selector")
    
    # Initialize the model interaction class
    llm = LLMChat(model_name)

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, total_items)

    for i in range(start_index, end_index):
        row = dataset[i]['data']

        # Display question and options
        display_question_options(row, i)

        if f"model_answer_{i}" not in st.session_state:
            reset_session_state(i)

        if st.button("Answer from Model", key=f"answer_{i}"):
            with st.spinner('Running model to get the answer...'):
                options = []
                for key, value in row["Options"].items():
                    options.append(value)
                first_line, explanation = llm.generate_answer(row['Question'], options)
                st.session_state[f"model_answer_{i}"] = f"**Model's Answer:** {first_line}\n\n**Explanation:** {explanation}"

        if st.session_state.get(f"model_answer_{i}"):
            st.write(st.session_state[f"model_answer_{i}"])

        explain_question = st.text_input(f"Enter your question for explanation (Q{i+1})", key=f"explain_input_{i}")
        if st.button("Explain", key=f"explain_button_{i}"):
            if explain_question:
                with st.spinner('Getting explanation from model...'):
                    explanation = llm.generate_explanation(explain_question)
                    st.write(f"**Explanation:** {explanation}")

        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    main()

