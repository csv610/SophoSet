from datasets import load_dataset

def load_data(subset, split):
    ds = load_dataset("truthfulqa/truthful_qa", subset)
    return ds[split]

def main():
    print("Dataset: TruthfulQA")

    subset = 'generation'  # You can change this to 'multiple_choice' if needed
    split = 'validation'

    dataset = load_data(subset, split)
    
    total_items = len(dataset)

    for i in range(total_items):
        row = dataset[i]
        print(f"\nQuestion {i + 1}:")
        print(f"Question: {row['question']}")

        if subset == "generation":
            print("Correct Answers:")
            for answer in row['correct_answers']:
                print(f"- {answer}")
            
            print("\nBest Answer:")
            print(row['best_answer'])

            print("\nIncorrect Answers:")
            for answer in row['incorrect_answers']:
                print(f"- {answer}")

        if subset == "multiple_choice":
            print("mc1 targets:")
            for choice in row['mc1_targets']['choices']:
                print(f"- {choice}")
            print(f"labels: {row['mc1_targets']['labels']}")

            print("\nmc2 targets:")
            for choice in row['mc2_targets']['choices']:
                print(f"- {choice}")
            print(f"labels: {row['mc2_targets']['labels']}")

        print("-" * 30)

if __name__ == "__main__":
    main()
