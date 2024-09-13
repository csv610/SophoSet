pg
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
from langchain.llms import OllamaLLM

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings
    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

st.title("Image Analysis with Ollama")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        pil_image = Image.open(uploaded_file)
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

        # Convert image to base64
        image_b64 = convert_to_base64(pil_image)

        # Initialize Ollama
        llm = OllamaLLM(model="bakllava", temperature=0.1)

        # Text input for user question
        user_question = st.text_input("Ask a question about the image:")

        if user_question:
            with st.spinner("Analyzing..."):
                # Invoke Ollama with the image context and user question
                response = llm.invoke(user_question, images=[image_b64])

            # Display the response
            st.subheader("Analysis Result:")
            st.write(response)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

else:
    st.write("Please upload an image to begin.")
