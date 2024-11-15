import argparse

from model_query_base import ModelQueryBase

class RealWorldQA_Query(ModelQueryBase):
    def __init__(self, model_name):
        self.dataset_name = "visheratin/realworldqa"
        super().__init__(self.dataset_name, model_name)

    def split_text(text: str):
        """Split the text into question and options."""
        if "Please" in text:
            text = text[:text.rfind("Please")].strip()

        parts = text.split("?", 1)
        if len(parts) == 2:
            question = parts[0] + "?"
            options = re.split(r'(?=\b[A-Z]\.\s)', parts[1].strip())
            options = [opt.strip() for opt in options if opt.strip()]
        else:
            question = text
            options = []

        return question, options

    def process_question(self, row):
        question = row['question']

        question, options = split_text(row['question'])

        image = None
        if row['image']:
           image = row['image']
        return self.get_response(question, options, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RealworldQA dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None,
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.2",
                        help="Name of the model to use for processing.")
    args = parser.parse_args()

    eval = RealWorldQA_Query(model_name=args.model_name)
    eval.run(nsamples=args.nsamples)

