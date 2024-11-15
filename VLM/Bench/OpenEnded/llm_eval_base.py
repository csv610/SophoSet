import logging
import random
import argparse
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import time

from tqdm import tqdm
from datasets import load_dataset
from datasets import get_dataset_config_names
from datasets import get_dataset_split_names

from utils import load_data
from utils import gen_question_id
from utils import save_results
from utils import get_sample_indices
from llm_model import load_llm_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMEvalBase(ABC):
    def __init__(self, dataset_name, model_name):
        self.dataset_name = dataset_name
        self.model_name = model_name
        try:
            self.llm = load_llm_model(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def ask_llm(self, question, choices, retries=3, wait=5):
        print( question )
        return "Return"
        for attempt in range(retries):
            try:
                response = self.llm.get_answer(question, choices)
                return response['answer']
            except ConnectionError as e:
                logger.error(f"Connection error while getting answer from model (attempt {attempt + 1}): {e}")
            except TimeoutError as e:
                logger.error(f"Timeout error while getting answer from model (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error while getting answer from model (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                return "Error"

    def process_subset(self, subject, split, nsamples=None):
        dataset = load_data(self.dataset_name, subject, split)
        if dataset is None:
            logger.warning(f"Dataset for {subject} - {split} could not be loaded.")
            return None

        indices = get_sample_indices(dataset, nsamples)

        data = []
        desc = f"{split}" if subject is None else f"{subject}:{split}"
        for i in tqdm(indices, desc=desc, leave=False):
            try:
                id = gen_question_id(subject, split, i)
                answer = self.process_question(dataset[i])
                newdata = {'id': id, 'answer': answer}
                data.append(newdata)
            except Exception as e:
                logger.error(f"Error processing row {i} in {subject}:{split}: {e}")

        logger.info(f"Successfully processed {len(data)} rows for {subject}:{split}")
        return data

    def save_results(self, data):
        save_results(data, self.dataset_name, self.model_name)

    def process_dataset(self, nsamples=None):
        logger.info("Starting dataset processing")

        subjects = get_dataset_config_names(self.dataset_name)
        if not subjects:
            logger.warning("No subjects found for the dataset. Exiting processing.")
            return

        frames = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for subject in subjects:
                splits =  get_dataset_split_names(self.dataset_name, subject)
                for split in splits:
                    futures.append(executor.submit(self.process_subset, subject, split, nsamples))
            
            for future in tqdm(futures, desc="Processing subsets and splits"):
                try:
                    data = future.result()
                    if data is not None:
                        frames.extend(data)
                    else:
                        logger.warning(f"Failed to load dataset for one of the subsets or splits")
                except Exception as e:
                    logger.error(f"An error occurred while processing a future: {e}")

        self.save_results(frames)

    @abstractmethod
    def process_question(self, row):
        pass

    def run(self, nsamples=None):
        self.process_dataset(nsamples=nsamples)

