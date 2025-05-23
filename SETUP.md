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

## 5. Ready to Go!

Your environment is now set up. You can proceed to run the project scripts:

- Adapt `prepare_custom_data.py` for your dataset and run it to generate `custom_dpo_data.jsonl`.
- Run `python run_dpo_training.py` to start the DPO fine-tuning process (ensure you have adequate GPU resources).
- Run `python test_finetuned_model.py` to test your fine-tuned model.

## Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:
```bash
deactivate
```
