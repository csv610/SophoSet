import streamlit as st
import requests
from datasets import load_dataset
from PIL import Image
from io import BytesIO

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

# Streamlit app
def main():
    st.title("MMLU Dataset")

    # Dictionary of categories and their corresponding subjects
    categories = {
        "STEM": [
            'abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry',
            'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security',
            'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
            'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
            'high_school_mathematics', 'high_school_physics', 'high_school_statistics',
            'machine_learning', 'medical_genetics', 'virology'
        ],
        "History and Social Sciences": [
            'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
            'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_us_history',
            'high_school_world_history', 'international_law', 'jurisprudence', 'prehistory',
            'sociology', 'us_foreign_policy', 'world_religions'
        ],
        "Medicine and Health Sciences": [
            'anatomy', 'clinical_knowledge', 'college_medicine', 'human_aging', 'human_sexuality',
            'medical_genetics', 'nutrition', 'professional_medicine', 'virology'
        ],
        "Philosophy and Ethics": [
            'business_ethics', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy'
        ],
        "Business and Management": [
            'econometrics', 'management', 'marketing', 'professional_accounting', 'public_relations'
        ],
        "Law and Political Science": [
            'international_law', 'jurisprudence', 'professional_law', 'security_studies', 'us_foreign_policy'
        ],
        "Psychology and Social Sciences": [
            'high_school_psychology', 'professional_psychology', 'sociology'
        ],
        "Miscellaneous": [
            'global_facts', 'miscellaneous'
        ]
    }

    # Sidebar for category and subject selection
    category = st.sidebar.selectbox("Select Category", list(categories.keys()))
    subject = st.sidebar.selectbox("Select Subject", categories[category])
    split = st.sidebar.selectbox("Select Split", ["test", "validation", "dev"])

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
    selected_page = st.sidebar.selectbox("Select Page Number", options=page_options)

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

        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    main()

