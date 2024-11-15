import argparse

from model_query_base import ModelQueryBase

class Cauldron_Query(ModelQueryBase):
    def __init__(self, model_name):
        self.dataset_name = "HuggingFaceM4/the_cauldron"

        super().__init__(self.dataset_name, model_name)

    def extract_data(data):
        questions = []
        options_list = []
        answers = []

        for item in data:
            user_text = item["user"]
            assistant_text = item["assistant"]
        
            # Extract question
            question_start = user_text.find("Question: ") + len("Question: ")
            question_end = user_text.find("\nChoices:")
            question = user_text[question_start:question_end].strip()
            questions.append(question)  # Append question to the list

            # Extract options
            options_start = user_text.find("\nChoices:\n") + len("\nChoices:\n")
            options_text = user_text[options_start:].split('\nAnswer with the letter.')[0]
            options = options_text.split("\n")
            options_list.append(options)  # Append options to the list

            # Extract answer
            answer_parts = assistant_text.split("Answer: ")
            if len(answer_parts) > 1:  # Check if the split was successful
                answer = answer_parts[1].strip()
            else:
                answer = "No answer provided."  # Handle the case where the answer is missing
            answers.append(answer)  # Append answer to the list

        return questions, options_list, answers

    def process_question(self, row):
        questions, options, answers = extract_data(row['texts'])
        images = row.get('images', [])
        return self.ask_llm(question, choices, images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Cauldron dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None,
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.2",
                        help="Name of the model to use for processing.")
    args = parser.parse_args()

    eval = Cauldron_Query(model_name=args.model_name)
    eval.run(nsamples=args.nsamples)

