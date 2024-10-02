import base64
import os
from io import BytesIO

import requests
from PIL import Image
import streamlit as st
from langchain_community.llms import Ollama

class LlavaChat:
    def __init__(self, model_name: str = "llava:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.llm = None
        self._initialize()  # Initialize in constructor

    def _convert_to_base64(self, pil_image: Image.Image) -> str:
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def _check_ollama_availability(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _initialize(self) -> None:
        if not self._check_ollama_availability():
            raise ConnectionError("Ollama server is not available. Please make sure it's running on http://localhost:11434")
        try:
            self.llm = Ollama(model=self.model_name)
        except Exception as e:
            raise RuntimeError(f"Error initializing Ollama: {str(e)}")

    def _process_image_input(self, image_input) -> str:  # Updated to accept any type
        from PIL import Image
        import requests
        from io import BytesIO

        if isinstance(image_input, str):
            # Check if it's a URL
            if image_input.startswith('http://') or image_input.startswith('https://'):
                response = requests.get(image_input)
                pil_image = Image.open(BytesIO(response.content))
            else:
                # Assume it's a local file
                pil_image = Image.open(image_input)
        elif isinstance(image_input, bytes):
            # If it's an encoded image (byte stream)
            pil_image = Image.open(BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            # Already a PIL image
            pil_image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}. Error: Unsupported type.")

        return self._convert_to_base64(pil_image)  # Convert to base64

    def get_answer(self, question: str, images: str | bytes | Image.Image | list = None) -> str:  # Default to None
        if not self.llm:
            self._initialize()
        
        try:
            images = self._prepare_images(images)  # Extracted image preparation logic
            response = self.llm.invoke(question, images=images)
            return response
        except Exception as e:
            raise RuntimeError(f"Error during image analysis: {str(e)}")

    def _prepare_images(self, images: str | bytes | Image.Image | list) -> list:  # New method for image preparation
        if images is None:
            return []
        
        if isinstance(images, str):  # Single image as string
            return [self._process_image_input(images)]
        elif isinstance(images, bytes):  # Single image as bytes
            return [base64.b64encode(images).decode("utf-8")]
        elif isinstance(images, Image.Image):  # Single image as PIL Image
            return [self._convert_to_base64(images)]
        elif isinstance(images, list):  # List of images
            return [self._process_image_input(img) for img in images]
        else:
            raise ValueError("Unsupported image input type.")

def main():
    st.set_page_config(page_title="LLaVA Chat", page_icon="🖼️")
    st.title("LLaVA Chat")

    llm = LlavaChat()

    input_type = st.radio("Choose input type:", ["File Upload", "Encoded Image String"])

    if input_type == "File Upload":
        image_input = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) 
    else:
        image_input = st.text_area("Paste your base64 encoded image string here:")

    image = None
    if image_input is not None:  # Check if an image has been uploaded
        image = Image.open(image_input)  # Read the image
        st.image(image, caption='Uploaded Image', use_column_width=True)  # Display the image

    user_question = st.text_input("Ask a question about the image", "What's in this image?")
        
    if st.button("Chat"):
        with st.spinner("Analyzing image..."):
            try:
                resp = llm.get_answer(user_question, image)
                st.write(resp)  
            except ConnectionError as e:
                st.error(str(e))
            except RuntimeError as e:
                st.error(str(e))
            except ValueError as e:
                st.error(str(e))


if __name__ == "__main__":
    main()
