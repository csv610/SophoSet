import logging
import pandas as pd
import random
import argparse

from tqdm import tqdm
from llm_chat import LLMChat
from typing import List, Tuple, Optional, Dict, Any
from datasets import load_dataset, Dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(subset: str, split: str) -> Optional[Dataset]:
    """Load a specific subset and split of the BigBenchHard dataset."""
    try:
        ds = load_dataset("maveriq/bigbenchhard", subset, trust_remote_code=True)
        logger.info(f"Successfully loaded dataset: {subset} - {split}")
        return ds[split]
    except Exception as e:
        logger.error(f"Error loading dataset {subset} - {split}: {str(e)}")
        return None

def split_text(text: str) -> Tuple[str, Optional[List[str]]]:
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

def process_subset(llm: LLMChat, subset: str, split: str, nsamples: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Process data for a specific subset and split."""
    dataset = load_data(subset, split)
    if dataset is None:
        logger.error(f"Failed to load dataset for {subset} - {split}")
        return None

    # Determine indices to process
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(range(len(dataset)), nsamples)
    else:
        indices = range(len(dataset))

    data = []
    logger.info(f"Starting processing for {subset}-{split}")

    model_name = llm.get_model_name()

    for i in tqdm(indices, desc=f"Processing {subset} {split}"):
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
#           'question': question, 
#           'options':  options, 
            'answer': row['target'],
            model_name : llm_answer
        })

    logger.info(f"Completed processing {len(data)} items for {subset} - {split}")
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
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting"
]

# List of splits
SPLITS = ["train"]

def process_dataset(model_name: str, nsamples: Optional[int] = None) -> None:
    logger.info("Starting dataset processing: Big Bench Hard")

    frames = []

    llm = LLMChat(model_name)
    logger.info(f"Initialized LLMChat with model: {model_name}")
    
    for subset in SUBSETS:
        for split in SPLITS:
            logger.info(f"Processing subset: {subset}, split: {split}")
            newframe = process_subset(llm, subset, split, nsamples)
            if newframe is not None:
                frames.append(newframe)
                logger.info(f"Successfully processed {subset} - {split}")
            else:
                logger.warning(f"Failed to process {subset} - {split}")
    
    if frames:
        final_frame = pd.concat(frames, ignore_index=True)
        filename = f"bigbenchhard_result_{model_name}"
        final_frame.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}jjkk")
    else:
        logger.error("No data processed. Unable to create CSV file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BigBenchHard dataset")
    parser.add_argument("-n", "--nsamples", type=int, default=None, 
                        help="Number of random samples to process per subset and split. If not provided, process all samples.")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.1", 
                        help="Name of the model to use. Default is llama3.1.")
    args = parser.parse_args()
    process_dataset(model_name=args.model_name, nsamples=args.nsamples)
