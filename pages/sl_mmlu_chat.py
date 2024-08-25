import streamlit as st
import requests
from datasets import load_dataset
from PIL import Image
from io import BytesIO
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(subject, split):
    ds = load_dataset("cais/mmlu", subject)
    return ds[split]

# Function to load image from a URL
def load_image_from_url(image_url):
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))

# Cache the model and define prompts to avoid reloading and redefining multiple times
@st.cache_resource()
def load_model_and_prompts(model_name):
    model = OllamaLLM(model=model_name)
    
    answer_prompt = ChatPromptTemplate.from_template("""
        Subject: {subject}

        Question: {question}

        Choices: 
        A) {choice_a}
        B) {choice_b}
        C) {choice_c}
        D) {choice_d}

        Answer: Let's carefully consider the subject matter and each of the provided choices before determining the most accurate answer. After evaluating the options, the answer is:
    """)
    
    explain_prompt_template = ChatPromptTemplate.from_template("""
        Subject: {subject}

        Question: {question}

        Let's think step by step and provide a brief explanation.
    """)
    
    return model, answer_prompt, explain_prompt_template

# Function to reset session state for the current index
def reset_session_state(index):
    st.session_state[f"model_answer_{index}"] = ""
    st.session_state[f"explain_input_{index}"] = ""

# Streamlit app
def main():
    st.title("MMLU Dataset")

    # List of available subjects
    subjects = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 
        'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 
        'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 
        'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 
        'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 
        'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 
        'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 
        'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 
        'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 
        'virology', 'world_religions'
    ]

    # Sidebar for subject and split selection
    subject = st.sidebar.selectbox("Select Subject", subjects, key="subject_selector")
    split = st.sidebar.selectbox("Select Split", ["test", "validation", "dev"], key="split_selector")
    
    # Model selection
    model_name = st.sidebar.selectbox("Select Model", ["llama3.1", "llama2", "gpt-3.5-turbo"], key="model_selector")
    model, answer_prompt, explain_prompt_template = load_model_and_prompts(model_name)
    
    # Load dataset
    dataset = load_data(subject, split)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Select number of items per page
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    
    # Calculate total pages
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + (total_items % num_items_per_page > 0)

    # Page selection box
    page_options = list(range(1, total_pages + 1))
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options, key="page_selector")

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, total_items)

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question: {i + 1}")

        # Display instruction
        st.write(f"**Question:** {row['question']}")

        # Display choices with (A), (B), (C), etc.
        choices = row['choices']
        choice_labels = ["(A)", "(B)", "(C)", "(D)", "(E)"]
        formatted_choices = [f"{label} {choice}" for label, choice in zip(choice_labels, choices)]
        st.write("**Choices:**")
        for choice in formatted_choices:
            st.write(choice)

        # Map the correct answer index to the corresponding letter
        correct_answer = choice_labels[row['answer']].strip("()")
        st.write(f"**Answer:** {correct_answer}")

        # Reset session state when a new question or subject is selected
        if f"model_answer_{i}" not in st.session_state:
            reset_session_state(i)

        # Button for "Answer from model"
        if st.button("Answer from Model", key=f"answer_{i}"):
            with st.spinner('Running model to get the answer...'):
                input_data = {
                    "subject": subject,
                    "question": row['question'],
                    "choice_a": choices[0],
                    "choice_b": choices[1],
                    "choice_c": choices[2],
                    "choice_d": choices[3],
                }
                model_response = (answer_prompt | model).invoke(input_data)

                # Splitting the response into the first answer and the explanation
                first_line, explanation = model_response.split("\n", 1)
                
                # Store the first line and explanation in session state
                st.session_state[f"model_answer_{i}"] = f"**Model's Answer:** {first_line}\n\n**Explanation:** {explanation}"

        # Display the stored answer and explanation
        if st.session_state.get(f"model_answer_{i}"):
            st.write(st.session_state[f"model_answer_{i}"])

        # Text input and button for explanation
        explain_question = st.text_input(f"Enter your question for explanation (Q{i+1})", key=f"explain_input_{i}")
        if st.button("Explain", key=f"explain_button_{i}"):
            if explain_question:
                with st.spinner('Getting explanation from model...'):
                    explanation = (explain_prompt_template | model).invoke({
                        "subject": subject,
                        "question": explain_question
                    })
                    st.write(f"**Explanation:** {explanation}")

        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    main()

