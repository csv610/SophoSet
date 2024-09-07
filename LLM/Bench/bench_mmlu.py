import pandas as pd
from datasets import load_dataset
from llm_chat import LLMChat
from tqdm import tqdm

# Load the dataset
def load_data(subject, split):
    try:
        dataset = load_dataset("cais/mmlu", subject)
        if split not in dataset:
            raise ValueError(f"Invalid split '{split}'. Available splits are: {', '.join(dataset.keys())}")
        return dataset[split]
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

# Process the dataset and return a pandas DataFrame
def process_subset(llm, subject, split):
    dataset = load_data(subject, split)
    if dataset is None:
        return pd.DataFrame()

    data = []
    for i, row in tqdm(enumerate(dataset), total=len(dataset), desc=f"Processing {subject} - {split}"):
        question = row['question']
        choices = row['choices']
        correct_answer = choice_labels[row['answer']].strip("()")

        try:
            llm_response = llm.get_answer(question, choices)
            llm_answer = llm_response['answer']
        except Exception as e:
            print(f"Error getting answer for question {i}: {str(e)}")
            llm_answer = "Error"

        data.append({
            "id": f"{subject}_{split}_{i + 1}",
            "Question": question,
            "Choices":  choices,
            "Answer":   correct_answer,
            "llama3.1": llm_answer
        })

    return pd.DataFrame(data)

def process_dataset():
    print("Dataset: MMLU")

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

    llm = LLMChat("llama3.1")

    splits = ["test", "validation", "dev"]

    # Iterate over all categories, subjects, and splits
    frames = []
    for category, subjects in categories.items():
        for subject in subjects:
            for split in splits:
                df = process_subset(llmsubject, split)
                frames.append(df)

    # Concatenate all dataframes into one
    final_df = pd.concat(frames, ignore_index=True)
    # Save the final dataframe to a CSV file
    final_df.to_csv("mmlu_result.csv", index=False)

if __name__ == "__main__":
    process_dataset()
