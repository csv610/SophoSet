import streamlit as st
from datasets import load_dataset
from PIL import Image
import os

st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data()
def load_data(split):
    model_name  = "Zhiqiang007/MathV360K"
    ds = load_dataset(model_name)
    ds = ds[split]
    return ds

# Function to load image from a file
def load_image_from_file(image_filename, image_folder="mathv360k_images"):
    image_path = os.path.join(image_folder, image_filename)
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

# Streamlit app
def main():
    st.title("MathV360K Dataset")

    # Load dataset
    split   = 'train'
    dataset = load_data(split)

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Select number of questions per page
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

        st.header(f"Question: {start_index + i + 1}")

        # Display image (from file)
        image = load_image_from_file(row['image'])
        st.image(image, caption=f"Qs: {start_index + i + 1}", use_column_width=True)

        # Display conversation
        if 'conversations' in row:
            display_conversation(row['conversations'])
        else:
            st.write("Conversation data not found.")

        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    main()

