# Docker Setup for Gemma Fine-tuning

This guide explains how to build and use the Docker container for fine-tuning the Gemma model. Using Docker provides a consistent and reproducible environment for running the fine-tuning scripts, especially on cloud instances or different machines.

## Prerequisites

- **Docker Installed:** Ensure Docker is installed on your system. See the [official Docker documentation](https://docs.docker.com/get-docker/) for installation instructions.
- **NVIDIA GPU Drivers (for GPU support):** If you plan to use GPUs for fine-tuning, ensure you have the appropriate NVIDIA drivers installed on your host machine. The Docker image includes the CUDA toolkit, but host drivers are necessary.
- **NVIDIA Container Toolkit:** For GPU access within Docker containers, you need to install the NVIDIA Container Toolkit. Follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## 1. Build the Docker Image

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
