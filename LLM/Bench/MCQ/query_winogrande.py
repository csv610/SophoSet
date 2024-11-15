import argparse
import re

from model_query_base import ModelQueryBase

class Winogrande_Eval(ModelQueryBase):
    def __init__(self, model_name):
        self.dataset_name = "automated-research-group/winogrande"
        super().__init__(self.dataset_name, model_name)

    def extract_question_options(self, text):
        # Regular expression to match the question and options
        pattern = r'^(.*?)\s*choose the option which is the most logical continuation of the text:\s*(.*)$'
        match = re.search(pattern, text, re.DOTALL)
    
        if match:
            question = match.group(1).strip()
            options_text = match.group(2).strip()
        
            # Extract individual options using regex
            options = re.findall(r'\d+\s*-\s*"(.*?)"', options_text)

            return question, options
        else:
            return None, None


    def process_question(self, row):
        question, options = self.extract_question_options(row['request'])
        return self.get_response(question, options)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Winogrande dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None,
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.2",
                        help="Name of the model to use for processing.")
    args = parser.parse_args()

    eval = Winogrande_Eval(model_name=args.model_name)
    eval.run(nsamples=args.nsamples)

