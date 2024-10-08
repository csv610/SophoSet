from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import time

class LLMChat:
    def __init__(self, model_name, temperature=0.5):
        self.model = OllamaLLM(model=model_name, temperature=temperature)
        self.mcq_prompt_template = """
            Question: {question}
            Choices:  {choices}

            Answer: Let's carefully consider the subject matter and each of the provided choices before determining the most accurate answer. Your response must include the final answer first and then the explanation.
        """
        self.explain_prompt_template = ChatPromptTemplate.from_template("""
            Question: {question}

            Let's think step by step and provide a brief explanation.
        """)

        self.default_prompt_template = ChatPromptTemplate.from_template("""
            Question: {question}
        """)

    def count_words(self, sentence: str) -> int:
        """Counts the number of words in a sentence."""
        words = sentence.split()
        return len(words)

    def get_answer(self, question, choices=None, cot_prompt=False):
        """Returns an answer. If choices are provided, it handles multiple choice; otherwise, it handles open-ended questions."""
        
        num_input_words = self.count_words(question)

        # If choices are provided, handle as an MCQ
        if choices:
            choices_str = "\n".join([f"{chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])
            num_input_words += sum(self.count_words(choice) for choice in choices)
            
            # Create the prompt with choices
            prompt = self.mcq_prompt_template.format(
                question=question,
                choices=choices_str
            )
        else:
            # If no choices, use the default or explanation prompt
            if cot_prompt:
                prompt = self.explain_prompt_template.format(question=question)
            else:
                prompt = self.default_prompt_template.format(question=question)

        # Measure response time
        t0 = time.time()
        response = (ChatPromptTemplate.from_template(prompt) | self.model).invoke({})
        t1 = time.time()

        num_output_words = self.count_words(response)
        elapsed_time = t1 - t0
        
        # If it's an MCQ, extract the answer letter
        if choices:
            answer_line = response.split("\n", 1)[0].strip()  # Get the first line
            answer = answer_line[0]  # Extract only the first character (A, B, C, etc.)
            explanation = response.split("\n", 1)[1] if "\n" in response else ""
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

import streamlit as st

@st.cache_resource
def load_llm_model(model_name: str):
    """Initialize and cache the LLM."""
    return LLMChat(model_name)

def ask_llm(llm, question, options, rowid):
    if st.button(f"LLM Answer:{rowid}"):
        with st.spinner("Processing ..."):
            try:
                response = llm.get_answer(question, options)
                st.write(f"Answer: {response['answer']}")
                st.write(f"Number of Input words: {response['num_input_words']}")
                st.write(f"Number of output  words: {response['num_output_words']}")
                st.write(f"Time: {response['response_time']}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if st.button(f"Ask Question: {rowid}"):
        user_question = st.text_area(f"Edit Question {rowid}", height=100)
        if user_question is not None:
            with st.spinner("Processing ..."):
                prompt = "Based on the Context of " + question + " Answer the User Question: " + user_question
                try:
                    response = llm.get_answer(prompt)
                    st.write(f"Answer: {response['answer']}")
                    st.write(f"Number of Input words: {response['num_input_words']}")
                    st.write(f"Number of output  words: {response['num_output_words']}")
                    st.write(f"Time: {response['response_time']}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

