import random
import logging
import argparse
import pandas as pd

from tqdm import tqdm
from llm_chat import LLMChat
from datasets import load_dataset

# Set up logging at the beginning of the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
def load_data(subject, split):
    try:
        dataset = load_dataset("cais/mmlu", subject)
        if split not in dataset:
            raise ValueError(f"Invalid split '{split}'. Available splits are: {', '.join(dataset.keys())}")
        return dataset[split]
    except Exception as e:
        logging.error(f"Error loading dataset for {subject} - {split}: {str(e)}")
        return None

# Process the dataset and return a pandas DataFrame
def process_subset(llm, subject, split, nsamples=None):
    dataset = load_data(subject, split)
    if dataset is None:
        return pd.DataFrame()

    logging.info(f"Processing subset: {subject} - {split}")
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))

    model_name = llm.get_model_name()

    data = []
    for i in tqdm(indices, total=len(indices), desc=f"Processing {subject} - {split}"):
        row = dataset[i]
        question = row['question']
        choices = row['choices']
        answer = row['answer']

        try:
            llm_response = llm.get_answer(question, choices)
            llm_answer = llm_response['answer']
        except Exception as e:
            logging.error(f"Error getting answer for question {i} in {subject} - {split}: {str(e)}")
            llm_answer = "Error"

        data.append({
            "id": f"{subject}_{split}_{i + 1}",
#           "Question": question,
#           "Choices":  choices,
            "answer":   answer,
            model_name: llm_answer
        })

    logging.info(f"Completed processing subset: {subject} - {split}")
    return pd.DataFrame(data)

def process_dataset(nsamples=None):
    logging.info(f"Starting MMLU dataset processing with nsamples={nsamples}")

    # Dictionary of categories and their corresponding subjects
    categories = {
        "STEM": [
            'abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry',
            'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security',
            'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
            'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
            'high_school_mathematics', 'high_school_physics', 'high_school_statistics',
            'machine_learning', 'medical_genetics', 'virology'
        ],
        "History and Social Sciences": [
            'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
            'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_us_history',
            'high_school_world_history', 'international_law', 'jurisprudence', 'prehistory',
            'sociology', 'us_foreign_policy', 'world_religions'
        ],
        "Medicine and Health Sciences": [
            'anatomy', 'clinical_knowledge', 'college_medicine', 'human_aging', 'human_sexuality',
            'medical_genetics', 'nutrition', 'professional_medicine', 'virology'
        ],
        "Philosophy and Ethics": [
            'business_ethics', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy'
        ],
        "Business and Management": [
            'econometrics', 'management', 'marketing', 'professional_accounting', 'public_relations'
        ],
        "Law and Political Science": [
            'international_law', 'jurisprudence', 'professional_law', 'security_studies', 'us_foreign_policy'
        ],
        "Psychology and Social Sciences": [
            'high_school_psychology', 'professional_psychology', 'sociology'
        ],
        "Miscellaneous": [
            'global_facts', 'miscellaneous'
        ]
    }

    model_name = "llama3.1"
    llm = LLMChat(model_name)
    logging.info("Initialized LLMChat with model: f"{model_name}")

    splits = ["test", "validation", "dev"]

    # Iterate over all categories, subjects, and splits
    frames = []
    for category, subjects in categories.items():
        for subject in subjects:
            for split in splits:
                df = process_subset(llm, subject, split, nsamples)
                frames.append(df)

    # Concatenate all dataframes into one
    final_df = pd.concat(frames, ignore_index=True)
    logging.info(f"Final dataset shape: {final_df.shape}")
    
    # Save the final dataframe to a CSV file
    final_df.to_csv("mmlu_result.csv", index=False)
    logging.info("Saved results to mmlu_result.csv")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Process MMLU dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, help="Number of samples to process. If not provided, process all samples.")
    args = parser.parse_args()

    logging.info(f"Starting MMLU dataset processing with nsamples={args.nsamples}")
    process_dataset(nsamples=args.nsamples)
    logging.info("MMLU dataset processing completed")
