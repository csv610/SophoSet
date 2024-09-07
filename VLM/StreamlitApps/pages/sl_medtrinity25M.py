from datasets import load_dataset
import streamlit as st

hf_token = "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"

@st.cache_resource
def load_data():
    ds = load_dataset("UCSC-VLAA/MedTrinity-25M", "25M_full", use_auth_token=hf_token)
    ds = ds['train']
    return ds

def main():
    st.title("Dataset: Med Trinity25M")
    st.divider()

    # Load the dataset using the cached function
    dataset = load_data()

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
        st.header(f"Question: {i + 1}")

        # Display question and answer
        image = row['image']
        caption = row['caption']
        st.image(image, caption=f"Qs:{i}")

        st.write(f"Caption: {caption}")

        st.divider()

if __name__ == "__main__":
    main()