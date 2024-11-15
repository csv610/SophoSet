import argparse

from model_query_base import ModelQueryBase

class BigBenchHard_Query(ModelQueryBase):
    def __init__(self, model_name):
        self.dataset_name = "maveriq/bigbenchhard"

        super().__init__(self.dataset_name, model_name)

    def split_text(self, text: str):
        """Split the input text into main sentence and options."""
        options_index = text.rfind("Options:")
    
        if options_index != -1:
           main_sentence = text[:options_index].strip()
           options = text[options_index + len("Options:"):].strip()
           clean_options = [opt.strip() for opt in options.split(")") if opt.strip()]
           formatted_options = [f"({chr(65 + i)}) {opt}" for i, opt in enumerate(clean_options)]
           return main_sentence, formatted_options
        else:
           return text, None

    def process_question(self, row):
        """Process a single question and return the answer."""
        try:
           question, options = self.split_text(row['input'])
           response = self.get_response(question, options)
           return response["answer"]
        except Exception as e:
           return "Error"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BigBenchHard dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None,
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.2",
                        help="Name of the model to use for processing.")
    args = parser.parse_args()

    eval = BigBenchHard_Query(model_name=args.model_name)
    eval.run(nsamples=args.nsamples)

