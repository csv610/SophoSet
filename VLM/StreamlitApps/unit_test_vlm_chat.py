import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import base64
from vlm_chat import LlavaChat  # Adjust the import based on your project structure

class TestLlavaChat(unittest.TestCase):

    @patch('vlm_chat.Ollama')  # Mocking the Ollama class
    @patch('vlm_chat.requests.get')  # Mocking requests.get
    def setUp(self, mock_get, mock_ollama):
        mock_get.return_value.status_code = 200  # Simulate a successful server response
        self.llava_chat = LlavaChat()

    @patch('vlm_chat.requests.get')
    def test_check_ollama_availability_success(self, mock_get):
        mock_get.return_value.status_code = 200
        self.assertTrue(self.llava_chat._check_ollama_availability())

    @patch('vlm_chat.requests.get')
    def test_check_ollama_availability_failure(self, mock_get):
        mock_get.return_value.status_code = 404
        self.assertFalse(self.llava_chat._check_ollama_availability())

    @patch('vlm_chat.requests.get')
    def test_initialize_success(self, mock_get):
        mock_get.return_value.status_code = 200
        self.llava_chat._initialize()
        self.assertIsNotNone(self.llava_chat.llm)

    @patch('vlm_chat.requests.get')
    def test_initialize_failure(self, mock_get):
        mock_get.return_value.status_code = 404
        with self.assertRaises(ConnectionError):
            self.llava_chat._initialize()

    @patch('vlm_chat.Image.open')
    def test_process_image_input_str(self, mock_open):
        image_input = "base64string"
        result = self.llava_chat._process_image_input(image_input)
        self.assertEqual(result, image_input)

    @patch('vlm_chat.base64.b64encode')
    def test_process_image_input_bytes(self, mock_b64encode):
        image_input = b'rawbytes'
        mock_b64encode.return_value = b'encodedbytes'
        result = self.llava_chat._process_image_input(image_input)
        self.assertEqual(result, 'encodedbytes')

    @patch('vlm_chat.Image.open')
    def test_process_image_input_pil_image(self, mock_open):
        mock_image = MagicMock(spec=Image.Image)
        mock_open.return_value = mock_image
        result = self.llava_chat._process_image_input(mock_image)
        self.assertIsInstance(result, str)  # Should return a base64 string

    def test_process_image_input_invalid(self):
        with self.assertRaises(ValueError):
            self.llava_chat._process_image_input(123)  # Invalid type

    @patch('vlm_chat.LlavaChat._prepare_images')
    @patch('vlm_chat.Ollama.invoke')
    def test_get_answer(self, mock_invoke, mock_prepare_images):
        mock_prepare_images.return_value = ['image1', 'image2']
        mock_invoke.return_value = "Answer"
        result = self.llava_chat.get_answer("What is this?", "image_input")
        self.assertEqual(result, "Answer")

    @patch('vlm_chat.LlavaChat._prepare_images')
    def test_get_answer_no_llm(self, mock_prepare_images):
        self.llava_chat.llm = None  # Simulate uninitialized LLM
        with self.assertRaises(ConnectionError):
            self.llava_chat.get_answer("What is this?", "image_input")

    def test_prepare_images_none(self):
        result = self.llava_chat._prepare_images(None)
        self.assertEqual(result, [])

    def test_prepare_images_single_string(self):
        result = self.llava_chat._prepare_images("base64string")
        self.assertEqual(len(result), 1)

    def test_prepare_images_single_bytes(self):
        result = self.llava_chat._prepare_images(b'rawbytes')
        self.assertEqual(len(result), 1)

    def test_prepare_images_single_pil_image(self):
        mock_image = MagicMock(spec=Image.Image)
        result = self.llava_chat._prepare_images(mock_image)
        self.assertEqual(len(result), 1)

    def test_prepare_images_list(self):
        result = self.llava_chat._prepare_images(["base64string", b'rawbytes'])
        self.assertEqual(len(result), 2)

    def test_prepare_images_invalid(self):
        with self.assertRaises(ValueError):
            self.llava_chat._prepare_images(123)  # Invalid type

if __name__ == '__main__':
    unittest.main()
