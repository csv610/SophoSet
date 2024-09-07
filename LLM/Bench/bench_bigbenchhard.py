import logging
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from llm_chat import LLMChat

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(subset, split):
    """Load a specific subset and split of the BigBenchHard dataset."""
    try:
        ds = load_dataset("maveriq/bigbenchhard", subset, trust_remote_code=True)
        logger.info(f"Successfully loaded dataset: {subset} - {split}")
        return ds[split]
    except Exception as e:
        logger.error(f"Error loading dataset {subset} - {split}: {str(e)}")
        return None

def split_text(text):
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

def process_subset(llm, subset, split):
    """Process data for a specific subset and split."""
    dataset = load_data(subset, split)
    if dataset is None:
        return None

    data = []
    logger.info(f"Processing {subset} - {split}")

    for i in tqdm(range(len(dataset)), desc=f"Processing {subset} {split}"):
        row = dataset[i]
        question, options = split_text(row['input'])

        try:
            llm_response = llm.get_answer(question, options)
            llm_answer = llm_response["answer"]
        except Exception as e:
            logger.error(f"Error processing {subset} {split} item {i+1}: {str(e)}")
            llm_answer = "Error"

        data.append({
            'id': f"{subset}_{split}_{i+1}",
            'question': question, 
            'options':  options, 
            'answer': row['target'],
            'llama3.1' : llm_answer
        })

    logger.info(f"Completed processing {subset} - {split}")
    return pd.DataFrame(data)

# List of subsets (tasks)
SUBSETS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five",
    "tracking_shuffled_objects_seven",
    "tracking_shuffled_objects_three",
    "web_of_lies",
    "word_sorting"
]

# List of splits
SPLITS = ["train", "validation", "test"]

def process_dataset():
    logger.info("Starting dataset processing: Big Bench Hard")

    frames = []
    llm = LLMChat("llama3.1")
    
    for subset in SUBSETS:
        for split in SPLITS:
            newframe = process_subset(llm, subset, split)
            if newframe is not None:
                frames.append(newframe)
            else:
                logger.warning(f"Failed to process {subset} - {split}")
    
    if frames:
        final_frame = pd.concat(frames, ignore_index=True)
        final_frame.to_csv("bigbenchhard_result.csv", index=False)
        logger.info("Data saved to bigbenchhard_result.csv")
    else:
        logger.error("No data processed. Unable to create CSV file.")

if __name__ == "__main__":
    process_dataset()
