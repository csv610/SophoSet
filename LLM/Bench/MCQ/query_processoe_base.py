import random
import argparse
import os
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import time

from tqdm import tqdm
from datasets import load_dataset
from datasets import get_dataset_config_names
from datasets import get_dataset_split_names

from utils import load_text_data, load_multimodal_data
from utils import gen_question_id
from utils import save_results
from utils import get_sample_indices
from llm_model import load_text_model, load_multimodal_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelProcessorBase(ABC):
    def __init__(self, dataset_name, text_model_name, multimodal_model_name):
        self.dataset_name = dataset_name
        self.text_model_name = text_model_name
        self.multimodal_model_name = multimodal_model_name
        self.current_model = None

    def load_model(self, model_name, model_type):
        if model_type == 'text':
            return load_text_model(model_name)
        elif model_type == 'multimodal':
            return load_multimodal_model(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def ask_model(self, question, choices=None, images=None, retries=3, wait=5):
        model_type = self.select_model(images)
        self.load_required_model(model_type)

        for attempt in range(retries):
            try:
                if model_type == 'text':
                    response = self.current_model.get_answer(question, choices)
                elif model_type == 'multimodal':
                    response = self.current_model.process_multimodal(question, choices, images)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

                return response.get('answer', 'No answer provided')
            except (ConnectionError, TimeoutError) as e:
                logger.error(f"{type(e).__name__} while getting answer from model (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.exception(f"Unexpected error while getting answer from model (attempt {attempt + 1}): {e}")

            if attempt < retries - 1:
                time.sleep(wait * (2 ** attempt))  # Exponential backoff
            else:
                return "Error"

    def load_required_model(self, model_type):
        if self.current_model is None or self.current_model_type != model_type:
            logger.info(f"Loading {model_type} model...")
            if model_type == 'text':
                self.current_model = self.load_model(self.text_model_name, 'text')
            elif model_type == 'multimodal':
                self.current_model = self.load_model(self.multimodal_model_name, 'multimodal')
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            self.current_model_type = model_type

    def select_model(self, images):
        if images is not None:
            return 'multimodal'
        else:
            return 'text'

    def process_subset(self, subject, split, nsamples=None):
        dataset = self.load_data(self.dataset_name, subject, split)
        if dataset is None:
            logger.warning(f"Dataset for {subject} - {split} could not be loaded.")
            return None

        indices = get_sample_indices(dataset, nsamples)
        logger.debug(f"Sample indices for {subject}:{split}: {indices}")

        data = []
        desc = f"{split}" if subject is None else f"{subject}:{split}"
        for i in tqdm(indices, desc=desc, leave=False):
            try:
                question_id = gen_question_id(subject, split, i)
                answer = self.process_question(dataset[i])
                processed_data = {'id': question_id, 'answer': answer}
                data.append(processed_data)
            except Exception as e:
                logger.error(f"Error processing row {i} in {subject}:{split}: {e}")
                logger.exception(e)

        logger.info(f"Successfully processed {len(data)} rows for {subject}:{split}")
        return data

    def save_results(self, data):
        save_results(data, self.dataset_name, self.text_model_name)

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
                splits = get_dataset_split_names(self.dataset_name, subject)
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
        model_type = self.select_model(row.get('image') if isinstance(row, dict) else None)
        self.load_required_model(model_type)
        if model_type == 'text':
            return self.process_text_question(row)
        elif model_type == 'multimodal':
            return self.process_multimodal_question(row)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def run(self, nsamples=None):
        self.process_dataset(nsamples=nsamples)

    @staticmethod
    def load_data(dataset_name, subject, split):
        # Automatically determine data type based on the dataset content
        dataset = load_multimodal_data(dataset_name, subject, split)
        if dataset and any(isinstance(item, dict) and 'image' in item for item in dataset):
            return dataset  # Multimodal dataset
        return load_text_data(dataset_name, subject, split)  # Default to text dataset

    @abstractmethod
    def process_text_question(self, row):
        pass

    @abstractmethod
    def process_multimodal_question(self, row):
        pass

