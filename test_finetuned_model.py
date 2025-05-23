import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- 0. Script Preamble and Prerequisites ---
print("--- Test Fine-tuned DPO Model Script ---")
print("Prerequisites:")
print("1. Ensure the fine-tuned model exists at the specified path.")
print("   This model is created after successfully running `run_dpo_training.py`.")
print("2. Required libraries (transformers, torch) should be installed.")
print("-" * 30)

# --- 1. Define Model Path ---
# This is the directory where the fine-tuned model and tokenizer were saved by DPOTrainer.
finetuned_model_path_base = "~/gemma3-4b-dpo-finetuned"
finetuned_model_path = os.path.expanduser(finetuned_model_path_base)

print(f"Attempting to load fine-tuned model from: {finetuned_model_path}")

# --- 2. Load Fine-tuned Model and Tokenizer ---
model = None
tokenizer = None

try:
    print("Loading fine-tuned model...")
    model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)
    print("Fine-tuned model loaded successfully!")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    print("Tokenizer loaded successfully!")

except OSError as e:
    print(f"\nOSError: Could not load the fine-tuned model or tokenizer from '{finetuned_model_path}'.")
    print(f"Details: {e}")
    print("Please ensure that:")
    print("  1. You have successfully run the `run_dpo_training.py` script.")
    print(f"  2. The path '{finetuned_model_path_base}' correctly points to the training output directory.")
    print("  3. The directory contains all necessary model files (e.g., 'pytorch_model.bin', 'config.json', 'tokenizer.json').")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the model or tokenizer: {e}")
    exit()

# --- 3. Set up for Inference ---
print("\n--- Setting up for Inference ---")
# Determine device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to device: {device}")

# Ensure tokenizer has a pad_token_id (common practice for generation)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"tokenizer.pad_token_id set to tokenizer.eos_token_id ({tokenizer.eos_token_id})")

# --- 4. Generate Text ---
print("\n--- Generating Text ---")

# Define a sample multi-turn prompt
# This prompt should ideally be in a similar style to the data used for DPO fine-tuning.
prompt_text = "User: Hi there!\nAssistant: Hello! How can I help you today?\nUser: I'm looking for a new laptop, something good for programming and occasional gaming."
# You can also try other prompts:
# prompt_text = "User: What is the capital of France?\nAssistant: The capital of France is Paris.\nUser: Can you tell me a fun fact about Paris?"
# prompt_text = "User: What is DPO?\nAssistant: DPO stands for Direct Preference Optimization.\nUser: Can you explain it like I'm five?"


print(f"\nOriginal Prompt:\n{prompt_text}")

# Encode the prompt
# The `.to(device)` ensures the input tensors are on the same device as the model.
try:
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
except Exception as e:
    print(f"Error during tokenization: {e}")
    exit()

# Generate text using the model
# These generation parameters can be tuned for different output styles.
print("\nGenerating response...")
try:
    # Ensure model is in evaluation mode (important if using dropout, batchnorm, etc., though less critical for pure inference here)
    model.eval()
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,       # Maximum number of new tokens to generate
            do_sample=True,           # Whether to use sampling; if False, uses greedy decoding
            top_k=50,                 # Restricts sampling to the top K most likely tokens
            top_p=0.95,               # Nucleus sampling: keeps the smallest set of tokens whose cumulative probability exceeds top_p
            temperature=0.7,          # Controls randomness: lower means less random, higher means more random
            pad_token_id=tokenizer.pad_token_id # Set pad_token_id
        )
except Exception as e:
    print(f"Error during model.generate(): {e}")
    exit()

# Decode the generated output
# `skip_special_tokens=True` removes tokens like [EOS], [PAD], etc., from the output string.
try:
    generated_text_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # The output includes the original prompt, so we can slice it if needed,
    # but for a conversational test, showing the full continuation is often useful.
    # To get only the newly generated part:
    # generated_text_only_new = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
except Exception as e:
    print(f"Error during decoding: {e}")
    exit()

# --- 5. Print Output ---
print("\n--- Generated Response (including prompt) ---")
print(generated_text_full)

# print("\n--- Generated Response (only new part) ---")
# print(generated_text_only_new)

print("\n--- Test Script Finished ---")
print("If the output looks reasonable, your DPO fine-tuned model is working!")
print("Consider trying different prompts and generation parameters.")
