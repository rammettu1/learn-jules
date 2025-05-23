# Docker Setup for Gemma Fine-tuning

This guide explains how to build and use the Docker container for fine-tuning the Gemma model. Using Docker provides a consistent and reproducible environment for running the fine-tuning scripts, especially on cloud instances or different machines.

## Prerequisites

- **Docker Installed:** Ensure Docker is installed on your system. See the [official Docker documentation](https://docs.docker.com/get-docker/) for installation instructions.
- **NVIDIA GPU Drivers (for GPU support):** If you plan to use GPUs for fine-tuning, ensure you have the appropriate NVIDIA drivers installed on your host machine. The Docker image includes the CUDA toolkit, but host drivers are necessary.
- **NVIDIA Container Toolkit:** For GPU access within Docker containers, you need to install the NVIDIA Container Toolkit. Follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## 1.a. Hugging Face Authentication for Gated Models (e.g., Gemma 3)

The Gemma 3 models are gated on Hugging Face. This means you must:
1.  **Accept Terms on Hugging Face Hub:** Go to the model's page (e.g., [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)) and ensure you've accepted any terms or agreements. You must be logged into your Hugging Face account.
2.  **Provide Authentication to Docker:** The Docker container needs access to your Hugging Face credentials to download the gated model. Here are a few ways to manage this:

    *   **Option 1: Log in Interactively Inside the Running Container (Recommended for Security)**
        After starting your container (e.g., with `docker run -it --rm --gpus all ... bash`):
        ```bash
        # Inside the container
        source .venv/bin/activate 
        huggingface-cli login
        ```
        Follow the prompts to enter your token. The token will be cached within the container (in the `appuser`'s home directory) for the current session and subsequent uses as long as the container persits (if not using `--rm`) or if you commit the container state.

    *   **Option 2: Use Hugging Face Token as an Environment Variable (Secure and Flexible)**
        You can pass your Hugging Face token as an environment variable when running the container. The `transformers` library will automatically detect and use it.
        ```bash
        docker run -it --rm --gpus all \
            -e HF_TOKEN="your_hugging_face_token_here" \
            -v /path/to/your/custom_data:/app/data \
            -v /path/to/your/output_models:/app/gemma3-4b-dpo-finetuned \
            gemma-finetune bash
        ```
        Replace `"your_hugging_face_token_here"` with your actual Hugging Face access token.

    *   **Option 3: Mount Local Hugging Face Cache/Configuration (Advanced)**
        You can mount your local Hugging Face configuration directory (which contains your token) into the container. The typical location is `~/.cache/huggingface/token` or `~/.huggingface/token`.
        ```bash
        # Example for Linux/macOS, token stored at ~/.cache/huggingface/token
        docker run -it --rm --gpus all \
            -v ${HOME}/.cache/huggingface:/home/appuser/.cache/huggingface \
            -v /path/to/your/custom_data:/app/data \
            -v /path/to/your/output_models:/app/gemma3-4b-dpo-finetuned \
            gemma-finetune bash
        ```
        This makes your local token available inside the container. Ensure the path `/home/appuser/.cache/huggingface` is the correct Hugging Face cache directory for the `appuser` inside the container.

    *   **Option 4: Pass Token as a Build Argument (Least Recommended for Security)**
        You can pass the token during the `docker build` step and have the `Dockerfile` log in.
        **Dockerfile change:**
        ```Dockerfile
        # ... (other parts of Dockerfile)
        ARG HF_TOKEN
        ENV HF_TOKEN=${HF_TOKEN}
        # Optionally, run login, though env var might be enough
        # RUN if [ -n "$HF_TOKEN" ]; then . .venv/bin/activate && huggingface-cli login --token $HF_TOKEN; fi
        # ...
        ```
        **Build command:**
        ```bash
        docker build --build-arg HF_TOKEN="your_hugging_face_token_here" -t gemma-finetune .
        ```
        **Caution:** This method embeds your token into the Docker image layer, which can be a security risk if the image is shared or pushed to a public registry.

Choose the method that best suits your security needs and workflow. For most users, **Option 2 (run-time environment variable)** or **Option 1 (interactive login)** are recommended.

## 1.b. Build the Docker Image

Navigate to the project's root directory (where the `Dockerfile` is located) and run the following command to build the Docker image:

```bash
docker build -t gemma-finetune .
```
This command will build an image named `gemma-finetune` using the instructions in the `Dockerfile`. The process might take some time as it downloads the base image and installs dependencies.

## 2. Running the Docker Container

Once the image is built, you can run a container based on it.

### Interactive Session (Recommended for Development/Debugging)

To start an interactive bash session within the container with GPU access:

```bash
docker run -it --rm --gpus all \
    -v /path/to/your/custom_data:/app/data \
    -v /path/to/your/output_models:/app/gemma3-4b-dpo-finetuned \
    gemma-finetune bash
```

**Explanation of flags:**
- `-it`: Runs the container in interactive mode with a pseudo-TTY.
- `--rm`: Automatically removes the container when it exits.
- `--gpus all`: Makes all available GPUs accessible to the container.
- `-v /path/to/your/custom_data:/app/data`: Mounts your local directory containing custom conversation data (e.g., where you'll place or generate `custom_dpo_data.jsonl`) to `/app/data` inside the container. **Replace `/path/to/your/custom_data` with the actual path on your host machine.**
- `-v /path/to/your/output_models:/app/gemma3-4b-dpo-finetuned`: Mounts a local directory to store the fine-tuned model output to `/app/gemma3-4b-dpo-finetuned` inside the container. This ensures your trained model persists after the container stops. **Replace `/path/to/your/output_models` with the actual path on your host machine.**
- `gemma-finetune`: The name of the image to run.
- `bash`: The command to run inside the container (starts a bash shell).

Inside the container, you'll be in the `/app` directory as the `appuser`.
You'll first need to activate the virtual environment:
```bash
source .venv/bin/activate
```

Then you can:
1.  **Prepare Data:**
    - If your data preparation script needs to access files from the mounted `/app/data` volume, adjust paths in `prepare_custom_data.py` accordingly.
    - Run `python prepare_custom_data.py`. Ensure `custom_dpo_data.jsonl` is created in a location accessible by the training script (e.g., in `/app` or `/app/data` if you adjust paths).
2.  **Run Training:**
    - `python run_dpo_training.py`
    - The fine-tuned model will be saved to `/app/gemma3-4b-dpo-finetuned`, which is mounted to your host machine.
3.  **Test Model:**
    - `python test_finetuned_model.py`

### Running a Script Directly

You can also run a script directly. For example, to execute the training script (assuming your data is already prepared and accessible within the image or via mounts):

```bash
docker run --rm --gpus all \
    -v /path/to/your/custom_dpo_data.jsonl:/app/custom_dpo_data.jsonl \
    -v /path/to/your/output_models:/app/gemma3-4b-dpo-finetuned \
    gemma-finetune \
    bash -c "source .venv/bin/activate && python run_dpo_training.py"
```
Make sure `custom_dpo_data.jsonl` is correctly placed or generated. If `prepare_custom_data.py` generates it, you might need a multi-step command or run interactively first.

## Important Considerations

- **Data Paths:** Pay close attention to paths when mounting volumes and accessing data within the container. The scripts inside the container will see paths relative to the container's filesystem.
- **Resource Requirements:** Fine-tuning language models is resource-intensive. Ensure your Docker host has sufficient RAM, disk space, and GPU memory.
- **Customizing `prepare_custom_data.py`:** You will likely need to modify `prepare_custom_data.py` to load and process your specific dataset format from the mounted data volume (e.g., `/app/data`).
```
