import os
from trl import DPOConfig

# --- 1. Initialize DPOConfig ---
# This section demonstrates how to set up the DPOConfig object from the TRL library.
# These parameters control various aspects of the DPO training process.
# Users should adjust these settings based on their specific dataset,
# hardware (especially GPU memory), and training goals.

# Resolve the home directory for the output path
output_directory = os.path.expanduser("~/gemma3-4b-dpo-finetuned")

dpo_config = DPOConfig(
    # --- Filesystem ---
    output_dir=output_directory,              # Where to save the trained model and logs.

    # --- Training Hyperparameters ---
    num_train_epochs=1,                       # Number of times to iterate over the training dataset.
                                              # Start with 1-3 and tune based on validation performance.
    per_device_train_batch_size=1,            # Batch size per GPU. Reduce if you encounter Out-Of-Memory (OOM) errors.
                                              # Increase for potentially faster training if memory allows.
    gradient_accumulation_steps=4,            # Number of steps to accumulate gradients before performing an optimizer update.
                                              # Effective batch size will be per_device_train_batch_size * gradient_accumulation_steps.
                                              # Useful when GPU memory is limited and you want a larger effective batch size.
    learning_rate=5e-5,                       # The initial learning rate for the AdamW optimizer.
                                              # Common values are 5e-5, 3e-5, 2e-5. Tune based on training stability and convergence.
    warmup_ratio=0.1,                         # Proportion of training steps to perform linear learning rate warmup.
                                              # For example, 0.1 means the first 10% of training steps will have a linear warmup.
                                              # Helps stabilize training in the initial phase.
    lr_scheduler_type="cosine",               # Learning rate scheduler type (e.g., "linear", "cosine", "constant").
    # beta: The beta parameter for DPO loss. Recommended to be between 0.1 and 0.5.
    # We will leave it at the default (0.1) for now.
    # beta=0.1,

    # --- Logging and Saving ---
    logging_steps=10,                         # Log training metrics (e.g., loss) every X steps.
    save_steps=100,                           # Save a checkpoint of the model every X steps.
                                              # Adjust based on training duration and storage capacity.
    save_total_limit=2,                       # Only keep the last `save_total_limit` checkpoints. Older ones are deleted.

    # --- Mixed Precision and Performance ---
    bf16=True,                                # Whether to use bfloat16 mixed precision.
                                              # Requires Ampere or newer NVIDIA GPUs. Can significantly speed up training
                                              # and reduce memory usage with minimal impact on model quality.
                                              # If bf16 is not available, consider fp16=True (though bf16 is generally preferred if available).
    # fp16=False,                             # Set to True if bf16 is not available but you want to use fp16.

    # --- Sequence Lengths ---
    max_prompt_length=512,                    # Maximum length of the prompt sequence.
                                              # Sequences longer than this will be truncated.
                                              # Adjust based on your data and GPU memory.
    max_length=1024,                          # Maximum total sequence length after tokenization (prompt + completion).
                                              # Should be >= max_prompt_length.
                                              # Adjust based on your data and GPU memory.

    # --- Other Parameters ---
    # gradient_checkpointing=True,            # Enables gradient checkpointing to save memory at the cost of a ~20% slowdown.
                                              # Useful for training very large models or with limited VRAM.
    # optim="adamw_torch",                    # Optimizer to use.
    # remove_unused_columns=False,            # Set to False if you have other columns in your dataset you want to keep.
)

# --- 2. Print Configuration ---
# This will display the initialized DPOConfig object, allowing users to verify their settings.
print("--- DPO Configuration ---")
print(dpo_config)

print("\n--- Key DPOConfig Parameters ---")
print(f"Output Directory: {dpo_config.output_dir}")
print(f"Number of Training Epochs: {dpo_config.num_train_epochs}")
print(f"Per Device Train Batch Size: {dpo_config.per_device_train_batch_size}")
print(f"Gradient Accumulation Steps: {dpo_config.gradient_accumulation_steps}")
print(f"Effective Batch Size: {dpo_config.per_device_train_batch_size * dpo_config.gradient_accumulation_steps}")
print(f"Learning Rate: {dpo_config.learning_rate}")
print(f"BF16 Enabled: {dpo_config.bf16}")
print(f"Max Prompt Length: {dpo_config.max_prompt_length}")
print(f"Max Total Length: {dpo_config.max_length}")

print("\nNote: This script only demonstrates DPOConfig initialization.")
print("Adjust parameters based on your specific needs and hardware.")
