import streamlit as st
from datasets import load_dataset
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Utility function to load data
@st.cache_data()
def load_data(dataset_name, subject, split):
    ds = load_dataset(dataset_name, subject)
    return ds[split]

# Utility function to load the model and prompts
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

# Function to reset session state
def reset_session_state(index):
    st.session_state[f"model_answer_{index}"] = ""
    st.session_state[f"explain_input_{index}"] = ""

# Function to handle model interaction and display results
def handle_model_interaction(index, subject, question, choices, model, answer_prompt, explain_prompt_template):
    if st.button("Answer from Model", key=f"answer_{index}"):
        with st.spinner('Running model to get the answer...'):
            input_data = {
                "subject": subject,
                "question": question,
                "choice_a": choices[0],
                "choice_b": choices[1],
                "choice_c": choices[2],
                "choice_d": choices[3],
            }
            model_response = (answer_prompt | model).invoke(input_data)
            first_line, explanation = model_response.split("\n", 1)
            st.session_state[f"model_answer_{index}"] = f"**Model's Answer:** {first_line}\n\n**Explanation:** {explanation}"

    if st.session_state.get(f"model_answer_{index}"):
        st.write(st.session_state[f"model_answer_{index}"])

    explain_question = st.text_input(f"Enter your question for explanation (Q{index+1})", key=f"explain_input_{index}")
    if st.button("Explain", key=f"explain_button_{index}"):
        if explain_question:
            with st.spinner('Getting explanation from model...'):
                explanation = (explain_prompt_template | model).invoke({
                    "subject": subject,
                    "question": explain_question
                })
                st.write(f"**Explanation:** {explanation}")

# Function to display a question and its options
def display_question(index, row):
    st.header(f"Question: {index + 1}")
    st.write(f"**Question:** {row['question']}")
    choices = row['choices']
    choice_labels = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    formatted_choices = [f"{label} {choice}" for label, choice in zip(choice_labels, choices)]
    st.write("**Choices:**")
    for choice in formatted_choices:
        st.write(choice)
    correct_answer = choice_labels[row['answer']].strip("()")
    st.write(f"**Answer:** {correct_answer}")
    return choices

# Main app function
def main():
    st.title("MMLU Dataset")

    # Configuration options
    dataset_name = st.sidebar.text_input("Dataset Name", value="cais/mmlu")
    subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics']  # Example subjects, customize for your dataset
    subject = st.sidebar.selectbox("Select Subject", subjects)
    split = st.sidebar.selectbox("Select Split", ["test", "validation", "dev"])
    model_name = st.sidebar.selectbox("Select Model", ["llama3.1", "llama2", "gpt-3.5-turbo"])

    # Load model, prompts, and dataset
    model, answer_prompt, explain_prompt_template = load_model_and_prompts(model_name)
    dataset = load_data(dataset_name, subject, split)

    # Sidebar for navigation
    num_items_per_page = st.sidebar.slider("Select Number of Items per Page", min_value=1, max_value=10, value=5)
    total_items = len(dataset)
    total_pages = (total_items // num_items_per_page) + (total_items % num_items_per_page > 0)
    selected_page = st.sidebar.selectbox("Select Page Number", list(range(1, total_pages + 1)))

    # Display items for the selected page
    start_index = (selected_page - 1) * num_items_per_page
    end_index = min(start_index + num_items_per_page, total_items)

    for i in range(start_index, end_index):
        row = dataset[i]
        reset_session_state(i)
        choices = display_question(i, row)
        handle_model_interaction(i, subject, row['question'], choices, model, answer_prompt, explain_prompt_template)
        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    main()

