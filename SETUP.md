# Project Setup with uv

This guide explains how to set up the Python environment for this project using the `uv` package manager. `uv` is a fast Python package installer and resolver.

## 1. Install uv

If you don't have `uv` installed, you can install it by following the official instructions.

**For Linux and macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Alternatively, using pip (if you have Python and pip already):**
```bash
pip install uv
```

Verify the installation:
```bash
uv --version
```

## 2. Create a Virtual Environment

Navigate to the project's root directory (where `requirements.txt` is located) and create a virtual environment:

```bash
uv venv
```
This will create a virtual environment named `.venv` in the current directory.

## 3. Activate the Virtual Environment

Activate the newly created virtual environment:

**For Linux and macOS (bash/zsh):**
```bash
source .venv/bin/activate
```

**For Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```
Your terminal prompt should change to indicate that the virtual environment is active.

## 4. Install Dependencies

With the virtual environment activated, install the required packages using `uv` and the `requirements.txt` file:

```bash
uv pip install -r requirements.txt
```
`uv` will download and install all the specified packages.

## 4.a. Hugging Face Authentication (Important for Gemma 3)

The Gemma 3 models are gated, meaning you need to agree to their terms and conditions on the Hugging Face Hub and authenticate your environment to download them.

**1. Accept Model Terms:**
   - Visit the model page for Gemma 3 (e.g., [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)) or the specific Gemma 3 model you intend to use.
   - Ensure you are logged into your Hugging Face account.
   - If prompted, accept the terms and conditions for using the model.

**2. Install Hugging Face CLI:**
   If you haven't already, install the `huggingface-hub` CLI tool. With your virtual environment (`.venv`) activated:
   ```bash
   uv pip install huggingface-hub
   ```

**3. Log In using Hugging Face CLI:**
   Authenticate your environment by logging in with your Hugging Face account. You have two main options:

   *   **Interactive Login (Recommended):**
      ```bash
      huggingface-cli login
      ```
      This will prompt you to enter your Hugging Face token. Follow the instructions. Your token will be saved locally for future use by `transformers` and other Hugging Face libraries.

   *   **Using an Environment Variable (for non-interactive environments):**
      You can set the `HF_TOKEN` environment variable to your Hugging Face access token (one with at least read permissions).
      ```bash
      export HF_TOKEN="your_hugging_face_token_here"
      ```
      The scripts will automatically use this environment variable.

After completing these authentication steps, your environment will be able to download the Gemma 3 model.

## 4.b. Ready to Go!

Your environment is now set up. You can proceed to run the project scripts:

- Adapt `prepare_custom_data.py` for your dataset and run it to generate `custom_dpo_data.jsonl`.
- Run `python run_dpo_training.py` to start the DPO fine-tuning process (ensure you have adequate GPU resources).
- Run `python test_finetuned_model.py` to test your fine-tuned model.

## Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:
```bash
deactivate
```
