import streamlit as st
from langchain_community.llms import Ollama
from PIL import Image
import base64
from io import BytesIO
import requests

class LlavaChat:
    def __init__(self, model_name="llava:latest", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.llm = None

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

    def _initialize(self):
        if not self._check_ollama_availability():
            raise ConnectionError("Ollama server is not available. Please make sure it's running on http://localhost:11434")
        try:
            self.llm = Ollama(model=self.model_name)
        except Exception as e:
            raise RuntimeError(f"Error initializing Ollama: {str(e)}")

    def _process_image_input(self, image_input):
        if isinstance(image_input, str):
            # Assume it's already a base64 encoded string
            return image_input
        elif isinstance(image_input, bytes):
            # It's raw bytes, so we need to encode it
            return base64.b64encode(image_input).decode("utf-8")
        else:
            # Assume it's a file-like object
            try:
                pil_image = Image.open(image_input)
                return self._convert_to_base64(pil_image)
            except Exception as e:
                raise ValueError(f"Unsupported image input type: {type(image_input)}. Error: {str(e)}")

    def get_answer(self, question: str, image = None) -> str:
        if not self.llm:
            self._initialize()
        
        try:
            if image:
                image_b64 = self._process_image_input(image)
                response = self.llm.invoke(question, images=[image_b64])
            else:
                response = self.llm.invoke(question, images=[])
            return response
        except Exception as e:
            raise RuntimeError(f"Error during image analysis: {str(e)}")

def main():
    st.set_page_config(page_title="LLaVA Chat", page_icon="🖼️")
    st.title("LLaVA Chat")

    llava_chat = LlavaChat()

    input_type = st.radio("Choose input type:", ["File Upload", "Encoded Image String"])

    if input_type == "File Upload":
        image_input = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    else:
        image_input = st.text_area("Paste your base64 encoded image string here:")

    if image_input:
        if isinstance(image_input, BytesIO):  # File upload
            st.image(image_input, caption="Uploaded Image", use_column_width=True)
        elif isinstance(image_input, str):  # Encoded string
            try:
                decoded_image = base64.b64decode(image_input)
                st.image(decoded_image, caption="Decoded Image", use_column_width=True)
            except:
                st.error("Invalid base64 string. Please check your input.")
                return

        user_question = st.text_input("Ask a question about the image", "What's in this image?")
        
        if st.button("Chat"):
            with st.spinner("Analyzing image..."):
                try:
                    resp = llava_chat.get_answer(image_input, user_question)
                    st.write(resp)
                except ConnectionError as e:
                    st.error(str(e))
                except RuntimeError as e:
                    st.error(str(e))
                except ValueError as e:
                    st.error(str(e))

if __name__ == "__main__":
    main()
