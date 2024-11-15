import argparse
import re
import ast

from model_query_base import ModelQueryBase

class MedicalMeadowMedQA_Query(ModelQueryBase):
    def __init__(self, model_name):
        self.dataset_name = "medalpaca/medical_meadow_medqa"
        super().__init__(self.dataset_name, model_name)

    def extract_question_and_captions(text):
        match = re.search(r"\?\s*(\{.*\})", question)
        if match:
            question = question[:match.start()] + "?"
            choices_dict_str = match.group(1)
            choices_dict = ast.literal_eval(choices_dict_str)
            choices = list(choices_dict.values())
        else:
            choices = []

        return question,choices

    def process_question(self, row):
        text = row['input']
        question, choices = extract_question_and_captions(text)
        return self.get_response(question, choices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Medical Meadow MedQA dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None,
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.2",
                        help="Name of the model to use for processing.")
    args = parser.parse_args()

    eval = MedicalMeadowMedQA_Query(model_name=args.model_name)
    eval.run(nsamples=args.nsamples)

