# TorchRec Quickstart: Run in 5 Minutes

This quickstart provides a **self-contained, ready-to-run** TorchRec training example
that works on any cloud provider. No data downloads required!

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           TorchRec Quickstart Architecture                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                                  YOUR CODE                                       │  │
│   │                                                                                  │  │
│   │      train_torchrec_quickstart.py                                               │  │
│   │      ├─ DLRM Model (Deep Learning Recommendation Model)                         │  │
│   │      ├─ Synthetic Data Generator (no downloads needed!)                         │  │
│   │      ├─ Distributed Training Setup (works on 1 to N GPUs)                       │  │
│   │      └─ Checkpointing & Metrics                                                 │  │
│   │                                                                                  │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                              │
│                                          ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                               TORCHREC LIBRARY                                   │  │
│   │                                                                                  │  │
│   │      EmbeddingBagCollection        DistributedModelParallel                     │  │
│   │      ├─ Efficient embedding        ├─ Model-parallel sharding                   │  │
│   │      │   lookups                   ├─ Automatic shard planning                  │  │
│   │      └─ GPU-optimized              └─ NCCL collective ops                       │  │
│   │                                                                                  │  │
│   │      KeyedJaggedTensor             TrainPipelineSparseDist                      │  │
│   │      ├─ Sparse feature batches     ├─ Overlapped data loading                   │  │
│   │      └─ Variable-length inputs     └─ Pipelined embedding lookups               │  │
│   │                                                                                  │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                              │
│                                          ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                            CLOUD DEPLOYMENT                                      │  │
│   │                                                                                  │  │
│   │      ┌────────────┐    ┌────────────┐    ┌────────────┐                         │  │
│   │      │    AWS     │    │   Azure    │    │    GCP     │                         │  │
│   │      │   (EKS)    │    │   (AKS)    │    │   (GKE)    │                         │  │
│   │      └────────────┘    └────────────┘    └────────────┘                         │  │
│   │                                                                                  │  │
│   │      Same Dockerfile + Kubernetes manifest works everywhere!                    │  │
│   │                                                                                  │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## What's Included

| File | Description |
|------|-------------|
| `train_torchrec_quickstart.py` | Self-contained training script with DLRM model and synthetic data |
| `Dockerfile` | Container for cloud deployment |
| `kubernetes_job.yaml` | Kubernetes job manifest (works on EKS, AKS, GKE) |
| `requirements.txt` | Python dependencies |

## Quick Start

### Option 1: Run Locally (Single GPU)

```bash
# Install dependencies
pip install torch torchrec fbgemm-gpu

# Run training
python train_torchrec_quickstart.py
```

### Option 2: Run Locally (Multi-GPU)

```bash
# 4 GPUs on single machine
torchrun --nproc_per_node=4 train_torchrec_quickstart.py
```

### Option 3: Docker Container

```bash
# Build the container
docker build -t torchrec-quickstart .

# Run with GPU support
docker run --gpus all torchrec-quickstart train_torchrec_quickstart.py

# Multi-GPU
docker run --gpus all torchrec-quickstart -m torch.distributed.run \
    --nproc_per_node=4 train_torchrec_quickstart.py
```

### Option 4: Kubernetes (AWS/Azure/GCP)

```bash
# Deploy to your Kubernetes cluster
kubectl apply -f kubernetes_job.yaml

# Monitor training
kubectl logs -f job/torchrec-quickstart-training -n torchrec
```

## Cloud-Specific Commands

### AWS (EKS)

```bash
# Create cluster with GPU nodes
eksctl create cluster --name torchrec-cluster \
    --node-type p4d.24xlarge \
    --nodes 2 \
    --region us-east-1

# Deploy training job
kubectl apply -f kubernetes_job.yaml
```

### Azure (AKS)

```bash
# Create cluster with GPU nodes
az aks create --resource-group mygroup \
    --name torchrec-cluster \
    --node-vm-size Standard_ND96asr_v4 \
    --node-count 2

# Deploy training job
kubectl apply -f kubernetes_job.yaml
```

### Google Cloud (GKE)

```bash
# Create cluster with GPU nodes
gcloud container clusters create torchrec-cluster \
    --machine-type a2-highgpu-8g \
    --num-nodes 2 \
    --accelerator type=nvidia-tesla-a100,count=8

# Deploy training job
kubectl apply -f kubernetes_job.yaml
```

## The DLRM Model

This quickstart implements **DLRM (Deep Learning Recommendation Model)**, the industry-standard
architecture for recommendation systems:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DLRM Architecture                                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   Dense Features (13)              Sparse Features (26)                                 │
│        │                                │                                               │
│        ▼                                ▼                                               │
│   ┌─────────────┐              ┌───────────────────┐                                   │
│   │  Bottom MLP │              │ Embedding Tables  │                                   │
│   │  512→256→64 │              │ (Model Parallel)  │                                   │
│   └──────┬──────┘              └─────────┬─────────┘                                   │
│          │                               │                                              │
│          └──────────────┬────────────────┘                                              │
│                         │                                                               │
│                         ▼                                                               │
│              ┌─────────────────────┐                                                    │
│              │  Feature Interaction│                                                    │
│              │    (All Pairwise)   │                                                    │
│              └──────────┬──────────┘                                                    │
│                         │                                                               │
│                         ▼                                                               │
│              ┌─────────────────────┐                                                    │
│              │     Top MLP         │                                                    │
│              │  512→256→1 + Sigmoid│                                                    │
│              └──────────┬──────────┘                                                    │
│                         │                                                               │
│                         ▼                                                               │
│                  Click Probability                                                      │
│                      (0-1)                                                              │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Customizing the Quickstart

### Change Hyperparameters

```bash
python train_torchrec_quickstart.py \
    --batch_size 4096 \
    --embedding_dim 256 \
    --num_epochs 10 \
    --learning_rate 0.5
```

### Use Your Own Data

Replace the `SyntheticDataset` class with your own data loader:

```python
from your_data import CustomDataLoader

# In main():
train_loader = CustomDataLoader(batch_size=args.batch_size)
```

### Scale to Multiple Nodes

Update `kubernetes_job.yaml`:

```yaml
spec:
  parallelism: 4  # Number of nodes
  completions: 4  # Must match parallelism
```

## Performance Expectations

| Configuration | GPUs | Throughput | Time to Train |
|--------------|------|------------|---------------|
| Single GPU | 1x A100 | ~50K samples/sec | ~30 min |
| Single Node | 8x A100 | ~350K samples/sec | ~5 min |
| Multi-Node | 16x A100 | ~600K samples/sec | ~3 min |

*Benchmarks on synthetic data with default hyperparameters*

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:

```bash
python train_torchrec_quickstart.py --batch_size 512
```

### NCCL Timeout

Increase timeout and enable debug logging:

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=23
```

### Pod Scheduling Issues

Check GPU availability:

```bash
kubectl describe nodes | grep -A 10 "nvidia.com/gpu"
```

## Next Steps

1. **Real Data**: Replace synthetic data with [Criteo 1TB dataset](../golden_training/)
2. **Advanced Sharding**: Explore [custom sharding strategies](../sharding/)
3. **Production**: See cloud-specific guides in [aws/](../aws/), [azure/](../azure/), [gcp/](../gcp/)

## References

- [TorchRec Documentation](https://pytorch.org/torchrec/)
- [DLRM Paper](https://arxiv.org/abs/1906.00091)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/trainer/overview/)
