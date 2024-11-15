from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import time
import re
import logging

class PromptHandler:
    def __init__(self):
        self.mcq_prompt_template = """
Question: {question}
Choices:  {choices}
Answer: Let's carefully analyze the subject matter and evaluate each choice to determine the most accurate answer. 
        Your response must contain two parts:
            1. Label of the correct choice (e.g., A, B, C, D).
            2. A detailed explanation justifying why the selected choice is correct.
        Provide the answer in the following format:
            Answer: <Label>
            Explanation: <Your explanation here>
            """

        self.cot_prompt_template = ChatPromptTemplate.from_template("""
            Question: {question}

            Let's think step by step and provide a brief explanation.
        """)

        self.zeroshot_prompt_template = ChatPromptTemplate.from_template("""
            Question: {question}
        """)

    def get_mcq_prompt(self, question, choices):
        choices_str = "\n" + "\n".join([f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])
        return self.mcq_prompt_template.format(question=question, choices=choices_str)

    def get_cot_prompt(self, question):
        return self.cot_prompt_template.format(question=question)

    def get_zeroshot_prompt(self, question):
        return self.zeroshot_prompt_template.format(question=question)

    def get_prompt(self, question, choices=None, cot_prompt=False):
        if choices:
            return self.get_mcq_prompt(question, choices)
        elif cot_prompt:
            return self.get_cot_prompt(question)
        else:
            return self.get_zeroshot_prompt(question)

    def extract_label_and_explanation(self, text, num_options):
        # Generate valid labels based on the number of options (e.g., ['A', 'B', 'C', ...])
        valid_labels = [chr(65 + i) for i in range(num_options)]

        # Regular expressions to capture the answer label and explanation
        label_pattern = r"Answer: (\w)"  # Captures a single letter after 'Answer:'
        explanation_pattern = r"Explanation:\s*(.+)"  # Captures explanation after 'Explanation:'

        # Extract the label and explanation
        label_match = re.search(label_pattern, text)
        explanation_match = re.search(explanation_pattern, text, re.DOTALL)  # DOTALL allows multi-line explanations

        # Extracted values
        label = label_match.group(1) if label_match else None
        explanation = explanation_match.group(1).strip() if explanation_match else None

        # Check if the label is valid
        if label not in valid_labels:
            raise ValueError(f"Invalid label '{label}'. It should be one of {valid_labels}.")

        return {
            "label": label,
            "explanation": explanation
        }

class OllamaModel:
    def __init__(self, model_name, temperature = 0.5):
        self.model = OllamaLLM(model=model_name, temperature = temperature)
        self.prompt_handler = PromptHandler()

    def count_words(self, sentence: str) -> int:
        """Counts the number of words in a sentence."""
        words = sentence.split()
        return len(words)
    
    def get_answer(self, question, choices=None):
        """Returns an answer. If choices are provided, it handles multiple choice; otherwise, it handles open-ended questions."""
        
        num_input_words = self.count_words(question)
        if choices:
            num_input_words += sum(self.count_words(choice) for choice in choices)

        # Get the appropriate prompt from the prompt handler
        prompt = self.prompt_handler.get_prompt(question, choices)

        t0 = time.time()
        response = (ChatPromptTemplate.from_template(prompt) | self.model).invoke({})
        t1 = time.time()

        num_output_words = self.count_words(response)
        elapsed_time = t1 - t0
        
        # If it's an MCQ, extract the answer label and explanation
        if choices:
            num_options = len(choices)
            result = self.prompt_handler.extract_label_and_explanation(response, num_options)
            answer = result["label"]
            explanation = result["explanation"]
        else:
            # For non-MCQ, the answer is the full response
            answer = response
            explanation = None
        
        retval = {
            "num_input_words": num_input_words,
            "num_output_words": num_output_words,
            "response_time": elapsed_time,
            "answer": answer,
            "explanation": explanation
        }
        
        return retval

    def get_model_name(self) -> str:
        """Returns the name of the model being used."""
        return self.model.model

def load_llama_model(model_name: str) -> OllamaModel:
    llm = OllamaModel(model_name, temperature=0.0)
    return llm

if __name__ == "__main__":
     prompter = PromptHandler()
     llm      = OllamaModel("llama3.2", prompter)

     question = "who discovered radium?"
     options  = ["Gandhi", "Bose", "Madame Curie", "Feymann"]

     response = llm.get_answer(question, options)
     print(response['answer'])
     print(response['explanation'])




