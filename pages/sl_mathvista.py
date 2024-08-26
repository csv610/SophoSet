import streamlit as st
from datasets import load_dataset

st.set_page_config(layout="wide")

def explore_data():
    # Set the title of the app
    st.title("Dataset: MathVista")
    st.divider()

    st.sidebar.title("Navigation")

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

    for i in range(start_index, end_index):
        row = dataset[i]
        st.write(f"**Question {i+1}:**")
        st.write(row['question'])

        image = row['decoded_image']
        st.image(image, caption=f"Qs:{i}")

        st.write(f"**Choices:** {row['choices']}")

        st.write(f"**Answer:** {row['answer']}")
        st.divider()

if __name__ == "__main__":
    explore_data()

