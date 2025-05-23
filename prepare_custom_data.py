import datasets
import json

# --- 1. Generate Dummy Multi-Turn Conversation Data ---
# This section creates a few example multi-turn conversations.
# Each conversation has a "prompt" (the conversation history),
# a "chosen" (good) response, and a "rejected" (bad) response.

# Users should replace this dummy data generation with their actual
# data loading and preprocessing logic. For example, you might load
# conversations from a CSV file, a JSON file, or a database.
# The key is to transform your data into a list of dictionaries,
# where each dictionary has the keys "prompt", "chosen", and "rejected".

dummy_data = [
    {
        "prompt": "User: Hi there!\nAssistant: Hello! How can I help you today?\nUser: I'm looking for a new laptop.",
        "chosen": "Assistant: Great! What are your primary needs? For example, are you looking for something lightweight for travel, powerful for gaming, or a budget-friendly option for everyday tasks?",
        "rejected": "Assistant: Laptops are expensive."
    },
    {
        "prompt": "User: What is the capital of France?\nAssistant: The capital of France is Paris.\nUser: Can you tell me a fun fact about Paris?",
        "chosen": "Assistant: Certainly! One fun fact is that Paris is home to the world's largest art museum, the Louvre, which houses masterpieces like the Mona Lisa and the Venus de Milo.",
        "rejected": "Assistant: Paris is a city."
    },
    {
        "prompt": "User: I'm trying to learn Python. Any tips for beginners?\nAssistant: Welcome to the world of Python! It's a great language to start with.\nUser: What's a good first project?",
        "chosen": "Assistant: A good first project could be a simple calculator, a to-do list application, or a text-based adventure game. These projects help solidify basic concepts like variables, loops, and functions.",
        "rejected": "Assistant: Just read a book."
    },
    {
        "prompt": "User: What is DPO?\nAssistant: DPO stands for Direct Preference Optimization. It's a method to align language models with human preferences.\nUser: Can you tell me more about its advantages?",
        "chosen": "Assistant: One key advantage is that it's more stable and computationally less expensive than traditional RLHF methods, as it doesn't require training a separate reward model.",
        "rejected": "Assistant: I don't know much about DPO."
    }
]

# --- 2. Converts to Hugging Face Dataset ---
# This step converts the list of dictionaries into a Hugging Face Dataset object.
# This format is commonly used in Hugging Face libraries for training and evaluation.

# If your data is already in a dictionary of lists (e.g., {"prompt": [...], "chosen": [...], "rejected": [...]}),
# you can use `datasets.Dataset.from_dict()`.
# If your data is a list of dictionaries (like `dummy_data` above),
# `datasets.Dataset.from_list()` is more appropriate.
hf_dataset = datasets.Dataset.from_list(dummy_data)

# You can also load data directly from files using:
# - `datasets.load_dataset('json', data_files='your_file.jsonl')` for JSONL
# - `datasets.load_dataset('csv', data_files='your_file.csv')` for CSV
# After loading, you might need to rename columns or process the text
# to fit the "prompt", "chosen", "rejected" structure.
# For example, if your CSV has columns 'conversation_history', 'good_answer', 'bad_answer':
# loaded_dataset = datasets.load_dataset('csv', data_files='my_data.csv')
# def transform_row(example):
#     return {
#         "prompt": example["conversation_history"],
#         "chosen": example["good_answer"],
#         "rejected": example["bad_answer"]
#     }
# hf_dataset = loaded_dataset.map(transform_row)


# --- 3. Saves the Dataset ---
# The Hugging Face Dataset object is saved to a JSONL file.
# Each line in the file will be a JSON object representing one training sample.
output_filename = "custom_dpo_data.jsonl"

# The `to_json()` method saves the dataset in JSONL format.
hf_dataset.to_json(output_filename, orient="records", lines=True)

# --- 4. Includes Comments ---
# Comments have been added throughout the script to explain each step.

# --- 5. Prints a Confirmation ---
# A message is printed to confirm that the data has been generated and saved.
print(f"Successfully generated and saved custom DPO data to: {output_filename}")
print(f"Number of samples in the dataset: {len(hf_dataset)}")

# To verify the content, you can load and print the first few examples:
# loaded_dataset = datasets.load_dataset('json', data_files=output_filename, split='train')
# print("\nFirst few examples from the saved file:")
# for i in range(min(3, len(loaded_dataset))):
#     print(loaded_dataset[i])
