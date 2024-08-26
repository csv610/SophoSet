import streamlit as st
from datasets import load_dataset

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

# Streamlit app
def main():
    st.title("Dataset: MathVision")
    st.divider()

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Select subset
    split = st.sidebar.selectbox("Select Dataset Subset", options=['test', 'testmini'])

    # Load dataset
    dataset = load_data(split)
    
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

    for i in range(start_index, end_index):
        row = dataset[i]
        st.header(f"Question {i + 1}")

        st.write(f"**Question**: {row['question']}")

        if row['decoded_image']:
            st.image(row['decoded_image'], caption=f"Question {start_index + i + 1}", use_column_width=True)

        if row['options']:
            st.write(f"**Options:** {row['options']}")

        if row['answer']:
            st.write(f"**Answer**: {row['answer']}")

        st.divider()

if __name__ == "__main__":
    main()