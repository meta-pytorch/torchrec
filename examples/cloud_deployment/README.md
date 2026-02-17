# TorchRec Cloud Deployment Guide

This guide provides comprehensive examples for deploying TorchRec distributed training
on major cloud providers: **AWS**, **Microsoft Azure**, and **Google Cloud Platform (GCP)**.

## Cloud Deployment Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        TorchRec Cloud Deployment Architecture                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│                              ┌─────────────────────────┐                                │
│                              │  torchrun / Kubeflow    │                                │
│                              │   Training Operator     │                                │
│                              └───────────┬─────────────┘                                │
│                                          │                                              │
│            ┌─────────────────────────────┼─────────────────────────────┐                │
│            │                             │                             │                │
│            ▼                             ▼                             ▼                │
│   ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐       │
│   │      AWS        │           │     Azure       │           │      GCP        │       │
│   │   EKS / EC2     │           │   AKS / VMs     │           │   GKE / VMs     │       │
│   └────────┬────────┘           └────────┬────────┘           └────────┬────────┘       │
│            │                             │                             │                │
│            ▼                             ▼                             ▼                │
│   ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐       │
│   │ p4d.24xlarge    │           │ ND96asr_v4      │           │ a2-highgpu-8g   │       │
│   │ 8x A100 (40GB)  │           │ 8x A100 (40GB)  │           │ 8x A100 (40GB)  │       │
│   │ 96 vCPUs        │           │ 96 vCPUs        │           │ 96 vCPUs        │       │
│   │ 1.1TB RAM       │           │ 900GB RAM       │           │ 1.3TB RAM       │       │
│   └────────┬────────┘           └────────┬────────┘           └────────┬────────┘       │
│            │                             │                             │                │
│            └─────────────────────────────┼─────────────────────────────┘                │
│                                          │                                              │
│                                          ▼                                              │
│                              ┌─────────────────────────┐                                │
│                              │   TorchRec Training     │                                │
│                              │  DistributedModelParallel│                               │
│                              │  TrainPipelineSparseDist │                               │
│                              └─────────────────────────┘                                │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Cloud Provider Comparison

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                     GPU Instance Comparison for TorchRec Training                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  Provider    Instance Type        GPUs           GPU Memory    Network        Cost/hr   │
│  ═══════════════════════════════════════════════════════════════════════════════════   │
│                                                                                         │
│  AWS         p4d.24xlarge        8x A100        40GB each     400 Gbps EFA   ~$32      │
│              p4de.24xlarge       8x A100        80GB each     400 Gbps EFA   ~$40      │
│              p5.48xlarge         8x H100        80GB each     3200 Gbps EFA  ~$98      │
│                                                                                         │
│  Azure       ND96asr_v4          8x A100        40GB each     200 Gbps IB    ~$27      │
│              ND96amsr_A100_v4    8x A100        80GB each     200 Gbps IB    ~$33      │
│              ND96isr_H100_v5     8x H100        80GB each     400 Gbps IB    ~$85      │
│                                                                                         │
│  GCP         a2-highgpu-8g       8x A100        40GB each     100 Gbps       ~$29      │
│              a2-ultragpu-8g      8x A100        80GB each     100 Gbps       ~$40      │
│              a3-highgpu-8g       8x H100        80GB each     200 Gbps       ~$80      │
│                                                                                         │
│  ═══════════════════════════════════════════════════════════════════════════════════   │
│                                                                                         │
│  Recommendation for TorchRec:                                                           │
│  • Training: p4d.24xlarge (AWS), ND96asr_v4 (Azure), a2-highgpu-8g (GCP)               │
│  • Large embeddings (>40GB): Use 80GB variants                                          │
│  • Multi-node: Prioritize high network bandwidth (EFA/InfiniBand)                       │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quickstart: Run TorchRec in 5 Minutes

**New to TorchRec?** Start with our [Quickstart Guide](quickstart/README.md) for a ready-to-run
example that works on any cloud provider!

```bash
# Build and run locally (single GPU)
cd quickstart && docker build -t torchrec-quickstart . && docker run --gpus all torchrec-quickstart train_torchrec_quickstart.py

# Or deploy to Kubernetes (multi-GPU, multi-node)
kubectl apply -f quickstart/kubernetes_job.yaml
```

The quickstart includes:
- ✅ Self-contained training script with synthetic data (no downloads needed!)
- ✅ Dockerfile for containerized deployment
- ✅ Kubernetes manifest for cloud deployment
- ✅ Works on AWS, Azure, and GCP

## Directory Structure

```text
cloud_deployment/
├── README.md                    # This file
├── quickstart/                  # ⭐ Start here!
│   ├── README.md               # Quickstart guide
│   ├── train_torchrec_quickstart.py  # Self-contained training script
│   ├── Dockerfile              # Container for cloud deployment
│   ├── kubernetes_job.yaml     # Kubernetes job manifest
│   └── requirements.txt        # Python dependencies
├── aws/
│   ├── README.md               # AWS-specific deployment guide
│   ├── eks_cluster.yaml        # EKS cluster configuration
│   ├── cloud_component.py     # TorchX component for AWS (legacy)
│   └── train_dlrm_aws.sh       # AWS training script
├── azure/
│   ├── README.md               # Azure-specific deployment guide
│   ├── aks_cluster.yaml        # AKS cluster configuration
│   ├── cloud_component.py     # TorchX component for Azure (legacy)
│   └── train_dlrm_azure.sh     # Azure training script
└── gcp/
    ├── README.md               # GCP-specific deployment guide
    ├── gke_cluster.yaml        # GKE cluster configuration
    ├── cloud_component.py     # TorchX component for GCP (legacy)
    └── train_dlrm_gcp.sh       # GCP training script
```

## Quick Start

### Prerequisites

1. **PyTorch with torchrun** (included with PyTorch >= 1.9):
   ```bash
   pip install torch torchrec fbgemm-gpu
   ```

2. **Install Cloud CLI Tools**:
   ```bash
   # AWS
   pip install awscli boto3

   # Azure
   pip install azure-cli

   # GCP
   pip install google-cloud-sdk
   ```

3. **Configure Cloud Credentials**:
   ```bash
   # AWS
   aws configure

   # Azure
   az login

   # GCP
   gcloud auth login
   ```

### Deploy TorchRec Training

Choose your cloud provider:

| Provider | Guide | Quick Command |
|----------|-------|---------------|
| AWS | [aws/README.md](aws/README.md) | `torchrun --nnodes=2 --nproc_per_node=8 ...` |
| Azure | [azure/README.md](azure/README.md) | `torchrun --nnodes=2 --nproc_per_node=8 ...` |
| GCP | [gcp/README.md](gcp/README.md) | `torchrun --nnodes=2 --nproc_per_node=8 ...` |

## Training Flow on Cloud

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         TorchRec Cloud Training Workflow                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   1. SETUP PHASE                                                                        │
│   ══════════════                                                                        │
│                                                                                         │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   │  Provision  │───►│   Upload    │───►│  Configure  │───►│   Deploy    │             │
│   │  GPU Cluster│    │   Dataset   │    │  Kubernetes │    │    Job      │             │
│   │  (EKS/AKS/  │    │  (S3/Blob/  │    │  + torchrun │    │             │             │
│   │   GKE)      │    │   GCS)      │    │             │    │             │             │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                                         │
│   2. TRAINING PHASE                                                                     │
│   ═════════════════                                                                     │
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │                         Kubernetes Cluster                                       │  │
│   │  ┌───────────────────────────────────────────────────────────────────────────┐  │  │
│   │  │                    TorchRec Distributed Training                          │  │  │
│   │  │                                                                           │  │  │
│   │  │   Node 0 (Rank 0-7)           Node 1 (Rank 8-15)                          │  │  │
│   │  │  ┌──────────────────┐        ┌──────────────────┐                         │  │  │
│   │  │  │ ┌────┐ ┌────┐   │        │ ┌────┐ ┌────┐   │                          │  │  │
│   │  │  │ │GPU0│ │GPU1│...│◄──────►│ │GPU0│ │GPU1│...│   NCCL All-to-All       │  │  │
│   │  │  │ └────┘ └────┘   │  High  │ └────┘ └────┘   │   Communication          │  │  │
│   │  │  │ Embedding Shards│  BW    │ Embedding Shards│                          │  │  │
│   │  │  └──────────────────┘ NVLink└──────────────────┘                          │  │  │
│   │  │                       + EFA/IB                                            │  │  │
│   │  └───────────────────────────────────────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
│   3. MONITORING & CHECKPOINTING                                                         │
│   ══════════════════════════════                                                        │
│                                                                                         │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                                │
│   │  TensorBoard│    │ Checkpoint  │    │   Metrics   │                                │
│   │  (Training  │    │  to Cloud   │    │  Dashboard  │                                │
│   │   Curves)   │    │   Storage   │    │ (CloudWatch/│                                │
│   │             │    │             │    │  Stackdriver)│                               │
│   └─────────────┘    └─────────────┘    └─────────────┘                                │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Best Practices for Cloud Deployment

### 1. Data Storage Strategy

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           Data Storage Recommendations                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  Cloud      Raw Data          Preprocessed Data        Checkpoints                      │
│  ═══════════════════════════════════════════════════════════════════════════════       │
│                                                                                         │
│  AWS        S3 Standard       S3 + FSx for Lustre     S3 Standard                      │
│             (cold storage)    (high-throughput I/O)   (durable)                         │
│                                                                                         │
│  Azure      Blob Storage      Azure NetApp Files      Blob Storage                     │
│             (cold tier)       (NFS mount)             (hot tier)                        │
│                                                                                         │
│  GCP        Cloud Storage     Filestore               Cloud Storage                    │
│             (standard)        (high-scale tier)       (standard)                        │
│                                                                                         │
│  ═══════════════════════════════════════════════════════════════════════════════       │
│                                                                                         │
│  💡 TIP: For Criteo 1TB dataset, use parallel file systems (FSx/NetApp/Filestore)      │
│          to achieve >10 GB/s read throughput needed for large batch training           │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2. Network Configuration

- **Enable high-bandwidth networking**: EFA (AWS), InfiniBand (Azure), GPUDirect (GCP)
- **Place nodes in same availability zone** for lowest latency
- **Use placement groups** (AWS) or proximity placement groups (Azure) for multi-node

### 3. Cost Optimization

- Use **Spot/Preemptible instances** for fault-tolerant training with checkpointing
- **Right-size instances**: Start with smaller GPU counts, scale up as needed
- **Enable auto-scaling** for inference workloads

## Troubleshooting

| Issue | Solution |
|-------|----------|
| NCCL timeout | Increase `NCCL_IB_TIMEOUT`, check security groups |
| OOM on GPU | Reduce batch size, enable gradient checkpointing |
| Slow data loading | Use cloud-native parallel file systems |
| Job preemption | Enable checkpointing every N iterations |

## Related Examples

- [golden_training/](../golden_training/) - Reference DLRM training implementation
- [ray/](../ray/) - Ray cluster integration
- [nvt_dataloader/](../nvt_dataloader/) - NVTabular for GPU data loading

## References

- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/trainer/overview/)
- [TorchX Documentation](https://pytorch.org/torchx/) (legacy)
- [AWS Deep Learning Containers](https://github.com/aws/deep-learning-containers)
- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/)
- [Google Cloud AI Platform](https://cloud.google.com/ai-platform)
