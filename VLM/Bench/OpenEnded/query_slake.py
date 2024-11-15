import argparse
import os

from model_query_base import ModelQueryBase

class SLAKE_Query(ModelQueryBase):
    def __init__(self, model_name):
        self.dataset_name = "BoKelvin/SLAKE"
        self.image_directory = '.'
        super().__init__(self.dataset_name, model_name)

    def set_image_directory(self, img_dir):
        self.image_directory = img_dir

    def process_question(self, row):
        question = row['question']

        img_dir = self.image_directory

        image = None
        if row['img_name']:
            if not os.path.exists(img_dir):
                return
            image = os.path.join(img_dir, row['img_name'])

         return self.get_response(question, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SLAKE dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None,
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.2",
                        help="Name of the model to use for processing.")
    args = parser.parse_args()

    eval = SLAKE_Query(model_name=args.model_name)
    eval.run(nsamples=args.nsamples)

