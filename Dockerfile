# Use an official NVIDIA CUDA image as a parent image
# Ensure this CUDA version is compatible with the PyTorch version and your target hardware
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_VERSION=0.1.40 # Pin uv version for reproducibility

# Install system dependencies, Python, and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    git && \
    rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install uv (pinned version)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.cargo/bin/uv --version && \
    mv /root/.cargo/bin/uv /usr/local/bin/uv

# Create a working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies using uv (with --no-cache to reduce layer size)
# Using --system-site-packages temporarily for uv to find global python in venv creation by scripts
# but then installing into the venv context.
RUN uv venv --python python3.10 && \
    . .venv/bin/activate && \
    uv pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY prepare_custom_data.py .
COPY run_dpo_training.py .
COPY test_finetuned_model.py .
# If custom_dpo_data.jsonl is small and meant to be a default, copy it too.
# Otherwise, it should be mounted by the user.
# COPY custom_dpo_data.jsonl .

# Create a non-root user and group
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Switch to the non-root user
USER appuser
WORKDIR /app

# Ensure scripts are executable (if needed, though python scripts usually don't require +x directly)
# RUN chmod +x prepare_custom_data.py run_dpo_training.py test_finetuned_model.py

# Set up the entrypoint to activate the venv
# This allows running scripts directly like `docker run myimage python run_dpo_training.py`
# Users can also get a bash session with `docker run -it --rm --gpus all myimage bash`
# and then manually activate with `source .venv/bin/activate`
CMD ["bash"]
