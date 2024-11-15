import argparse

from model_query_base import ModelQueryBase

class OlympicArena_Eval(LLMEvalBase):
    def __init__(self, model_name):
        self.dataset_name = "GAIR/OlympicArena"
        super().__init__(self.dataset_name, model_name)

    def load_image(image_url):
        response = requests.get(image_url)
        return Image.open(BytesIO(response.content))

    def process_question(self, row):
        question = row['problem']

#        language = detect(question)
#        if language != 'en':
#           return

        images = []
        if row['figure_urls']:
            for url in row['figure_urls']:
                images.append(load_image(url))

        return self.get_response(question, None, images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Olympic Arena dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None,
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.2",
                        help="Name of the model to use for processing.")
    args = parser.parse_args()

    eval = OlympicArena_Query(model_name=args.model_name)
    eval.run(nsamples=args.nsamples)
