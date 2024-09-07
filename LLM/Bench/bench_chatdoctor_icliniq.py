import pandas as pd
from datasets import load_dataset
from llm_chat import LLMChat
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(split):
    try:
        ds = load_dataset("lavita/ChatDoctor-iCliniq")
        logger.info(f"Successfully loaded {split} split of the dataset")
        return ds[split]
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def process_dataset():
    logger.info("Starting process: Dataset: ChatDoctor iCliniq")

    llm = LLMChat("llama3.1")
    split = "train"

    dataset = load_data(split)
    if dataset is None:
        logger.error("Failed to load the dataset. Please try again later.")
        return

    results = []
    nsamples = len(dataset)
    
    logger.info(f"Processing {nsamples} samples")
    for i in tqdm(range(nsamples), desc="Processing samples"):
        row = dataset[i]
        question = row['input']
        answer = row['answer_icliniq']

        try:
            llm_response = llm.get_answer(question)
            llm_answer = llm_response['response']
        except Exception as e:
            logger.error(f"Error processing question {i}: {str(e)}")
            llm_answer = "Error"
        
        results.append({
            'id': f"{split}_{i+1}",
            'Question': question,
            'Answer': answer,
            'llama3.1': llm_answer
        })

    # Create a pandas DataFrame
    df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    output_file = 'chatdoctor_icliniq_results.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    process_dataset()

