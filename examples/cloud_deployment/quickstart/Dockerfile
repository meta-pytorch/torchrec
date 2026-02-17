# TorchRec Quickstart Dockerfile
# Ready-to-run container for TorchRec distributed training on any cloud provider
#
# Build: docker build -t torchrec-quickstart .
# Run:   docker run --gpus all torchrec-quickstart
#
# Distributed training:
#   Single node:  docker run --gpus all torchrec-quickstart \
#                   torchrun --nproc_per_node=4 train_torchrec_quickstart.py
#   Multi-node:   See kubernetes_job.yaml for Kubernetes deployment

FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL maintainer="TorchRec Team"
LABEL description="TorchRec distributed training quickstart container"
LABEL version="1.0"

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace/torchrec

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train_torchrec_quickstart.py .

# Environment variables for distributed training
# NCCL settings (cloud-agnostic defaults; override per-provider via K8s env)
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_TIMEOUT=22

# Default command: show usage
CMD ["python", "-c", "print('TorchRec Quickstart Container\\n\\nUsage:\\n  Single GPU:  python train_torchrec_quickstart.py\\n  Multi-GPU:   torchrun --nproc_per_node=<NUM_GPUS> train_torchrec_quickstart.py\\n  Multi-Node:  torchrun --nnodes=<N> --nproc_per_node=<GPUS> --rdzv_backend=c10d --rdzv_endpoint=<MASTER_IP>:29500 train_torchrec_quickstart.py')"]

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import torchrec; print('OK')" || exit 1

# Expose port for distributed training rendezvous
EXPOSE 29500

# Default entrypoint for training
ENTRYPOINT ["python"]
