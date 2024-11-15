import argparse

from model_query_base import ModelQueryBase

class MathV360K_Query(ModelQueryBase):
    def __init__(self, model_name):
        self.dataset_name = "Zhiqiang007/MathV360K"
        self.image_folder = None

        super().__init__(self.dataset_name, model_name)

    def set_image_folder(imdir):
        self.image_folder = imdir
  
    def extract_data(conversation_list):
        question = None
        answer = None
        options = None

        for entry in conversation_list:
            if entry["from"] == "human":
               # Extract question and options from the human's message
               if "Question:" in entry["value"]:
                   question_part = entry["value"].split("Question:")[-1].strip()
                   question, options_part = question_part.split("Choices:") if "Choices:" in question_part else (question_part, None)
               if options_part:
                   options = [opt.strip() for opt in options_part.split("\n") if opt.strip()]  # Parse options

            elif entry["from"] == "gpt":
                # Extract answer from the GPT's message
                answer = entry["value"].strip()

            return question, options, answer 


    def process_question(self, row):
        question, options, answer = extract_data(row['conversations'])
        # Check if the image folder exists
        if not os.path.exists(image_folder):
            return  # Exit the function if the folder does not exist

        image = os.path.join(image_folder, row['image'])
        return self.get_response(question, options, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MathV360K  dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None,
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.2",
                        help="Name of the model to use for processing.")
    args = parser.parse_args()

    eval = MathV360K_Query(model_name=args.model_name)
    eval.run(nsamples=args.nsamples)

