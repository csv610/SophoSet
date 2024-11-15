import argparse

from model_query_base import ModelQueryBase

class Ai2d_Query(ModelQueryBase):
    def __init__(self, model_name):
        self.dataset_name = "lmms-lab/ai2d"
        super().__init__(self.dataset_name, model_name)

    def process_question(self, row):
        question = row['question']
        options  = row['options']
        image    = row['image']
        return self.get_response(question, options, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Ai2d dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None,
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.2",
                        help="Name of the model to use for processing.")
    args = parser.parse_args()

    eval = Ai2d_Query(model_name=args.model_name)
    eval.run(nsamples=args.nsamples)

