import argparse

from model_query_base import ModelQueryBase

class MMMU_Query(ModelQueryBase):
    def __init__(self, model_name):
        self.dataset_name = "MMMU/MMMU"
        super().__init__(self.dataset_name, model_name)

    def process_question(self, row):
        question = row['question']

        images = []
        for j in range(1, 8):
            image_key = f"image_{j}"
            if row[image_key] is not None:
                image = row[image_key]
                if image:
                    images.append(image)
                
        options = row['options']
        return self.ask_llm(question, options, images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MMMU dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None,
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.2",
                        help="Name of the model to use for processing.")
    args = parser.parse_args()

    eval = MMMU_Query(model_name=args.model_name)
    eval.run(nsamples=args.nsamples)

