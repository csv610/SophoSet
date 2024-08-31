import pandas as pd
from datasets import load_dataset
from llm_chat import LLMChat
from tqdm import tqdm

def load_data(split='train'):
    try:
        ds = load_dataset("lavita/ChatDoctor-iCliniq")
        return ds[split]
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def main():
    print("Dataset: ChatDoctor iCliniq")

    dataset = load_data()
    if dataset is None:
        print("Failed to load the dataset. Please try again later.")
        return

    llm = LLMChat("llama3.1")

    results = []
    nsamples = len(dataset)
    
    # Use tqdm to create a progress bar
    for i in tqdm(range(nsamples), desc="Processing samples"):
        row = dataset[i]
        question = row['input']
        answer = row['answer_icliniq']
        llm_response = llm.get_answer(question)
        llm_answer = llm_response['response']
        
        results.append({
            'Question': question,
            'Correct Answer': answer,
            'llama3.1 Answer': llm_answer
        })

    # Create a pandas DataFrame
    df = pd.DataFrame(results)

    # Optionally, save the DataFrame to a CSV file
    df.to_csv('chatdoctor_icliniq_results.csv', index=False)
    print("\nResults saved to chatdoctor_icliniq_results.csv")

if __name__ == "__main__":
    main()

