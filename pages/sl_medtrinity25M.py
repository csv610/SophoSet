from datasets import load_dataset
from PIL import Image
import base64
from io import BytesIO

hf_token = "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"

# Convert image to base64 string
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def main():
    st.title("Med Trinity25M  Dataset")

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Load the dataset with permission using the Hugging Face token
    dataset = load_dataset("UCSC-VLAA/MedTrinity-25M", "25M_full", use_auth_token=hf_token)
    dataset = dataset['split']

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
        st.header(f"Question: {start_index + i + 1}")

        # Display question and answer
        image = row['image'])
        caption = row['caption'])
        st.image(image, caption=f"Qs:{i}")

        st.write(f"Caption: {caption}")

        st.markdown("---")  # Divider between items

if __name__ == "__main__":
    main()
