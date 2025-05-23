from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Define Model ID
model_id = "google/gemma-3-4b-it" # Using a smaller, but still capable Gemma model for faster download/loading

# --- Load Model ---
model = None
try:
    print(f"Loading model: {model_id}...")
    # 2. Load Model
    # Note: If you have limited RAM or no GPU, loading a large model like 4b might be slow or cause issues.
    # For very large models, consider device_map="auto" if you have multiple GPUs/enough RAM,
    # or BitsAndBytesConfig for quantization if you have limited resources.
    model = AutoModelForCausalLM.from_pretrained(model_id)
    print("Model loaded successfully!")
except ImportError:
    print("ImportError: One of the necessary libraries (transformers, torch, etc.) is not installed correctly.")
    print("Please ensure you have run: pip install transformers torch accelerate")
    exit()
except OSError as e:
    print(f"OSError: Could not load model. This might be due to a network issue or an incorrect model_id: {model_id}")
    print(f"Details: {e}")
    print("Please check your internet connection and the model ID on Hugging Face Hub.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")
    exit()

# --- Load Tokenizer ---
tokenizer = None
try:
    print(f"\nLoading tokenizer for: {model_id}...")
    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Tokenizer loaded successfully!")
except OSError as e:
    print(f"OSError: Could not load tokenizer. This might be due to a network issue or an incorrect model_id: {model_id}")
    print(f"Details: {e}")
    print("Please check your internet connection and the model ID on Hugging Face Hub.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the tokenizer: {e}")
    exit()

# --- Confirmation and Verification ---
if model and tokenizer:
    print("\n--- Model and Tokenizer Load Confirmation ---")
    print(f"Successfully loaded model: {model_id}")
    print(f"Successfully loaded tokenizer for: {model_id}")

    # 4. Optionally, print model.config
    print("\n--- Model Configuration (First 5 parameters) ---")
    try:
        # Displaying only a few key parameters from the config for brevity
        config_subset = {k: model.config.to_dict().get(k) for k in list(model.config.to_dict().keys())[:5]}
        for key, value in config_subset.items():
            print(f"{key}: {value}")
        if len(model.config.to_dict()) > 5:
            print("...") # Indicate that there are more parameters
    except Exception as e:
        print(f"Could not print model config: {e}")


    # 5. Tokenize a sample text and print input IDs
    sample_text = "Hello Gemma!"
    print(f"\n--- Tokenizer Verification ---")
    print(f"Tokenizing sample text: '{sample_text}'")
    try:
        inputs = tokenizer(sample_text, return_tensors="pt") # "pt" for PyTorch tensors
        print(f"Input IDs: {inputs['input_ids']}")
        # You can also decode to verify:
        # print(f"Decoded input IDs: {tokenizer.decode(inputs['input_ids'][0])}")
    except Exception as e:
        print(f"Could not tokenize sample text: {e}")

    print("\nScript finished successfully.")
else:
    print("\nModel or Tokenizer not loaded. Please check the error messages above.")
