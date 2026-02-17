# TorchRec on Google Cloud Platform

This guide provides step-by-step instructions for deploying TorchRec distributed training on Google Cloud Platform (GCP).

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           TorchRec on GCP Architecture                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                     Google Kubernetes Engine (GKE)                              │   │
│   │  ┌─────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │                    GPU Node Pool (a2-highgpu-8g)                        │   │   │
│   │  │                                                                         │   │   │
│   │  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │   │   │
│   │  │   │   Node 0    │    │   Node 1    │    │   Node 2    │   ...          │   │   │
│   │  │   │ 8x A100 GPU │    │ 8x A100 GPU │    │ 8x A100 GPU │                │   │   │
│   │  │   │ 96 vCPUs    │    │ 96 vCPUs    │    │ 96 vCPUs    │                │   │   │
│   │  │   │ 1.3TB RAM   │    │ 1.3TB RAM   │    │ 1.3TB RAM   │                │   │   │
│   │  │   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                │   │   │
│   │  │          │                  │                  │                       │   │   │
│   │  │          └──────────────────┼──────────────────┘                       │   │   │
│   │  │                             │                                          │   │   │
│   │  │                    ┌────────▼────────┐                                 │   │   │
│   │  │                    │  NVLink + VPC   │   100 Gbps network              │   │   │
│   │  │                    │   Networking    │   GPUDirect-TCPX                │   │   │
│   │  │                    └─────────────────┘                                 │   │   │
│   │  └─────────────────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                              │
│                    ┌─────────────────────┼─────────────────────┐                        │
│                    │                     │                     │                        │
│                    ▼                     ▼                     ▼                        │
│           ┌───────────────┐     ┌───────────────┐     ┌───────────────┐                 │
│           │Cloud Storage  │     │  Filestore    │     │ Cloud Monitor │                 │
│           │  (Raw Data)   │     │  (Fast I/O)   │     │   (Metrics)   │                 │
│           └───────────────┘     └───────────────┘     └───────────────┘                 │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Recommended Machine Types

| Machine Type | GPUs | GPU Memory | vCPUs | RAM | Network | Use Case |
|--------------|------|------------|-------|-----|---------|----------|
| a2-highgpu-8g | 8x A100 | 40GB | 96 | 1.3TB | 100 Gbps | Standard training |
| a2-ultragpu-8g | 8x A100 | 80GB | 96 | 1.3TB | 100 Gbps | Large embeddings |
| a3-highgpu-8g | 8x H100 | 80GB | 208 | 1.8TB | 200 Gbps | Maximum performance |

## Prerequisites

1. **gcloud CLI installed and configured**:
   ```bash
   gcloud auth login
   gcloud config set project your-project-id
   ```

2. **kubectl configured**:
   ```bash
   gcloud container clusters get-credentials torchrec-gke --zone us-central1-a
   ```

3. **Enable required APIs**:
   ```bash
   gcloud services enable container.googleapis.com
   gcloud services enable compute.googleapis.com
   gcloud services enable file.googleapis.com
   ```

## Step 1: Create GKE Cluster

```bash
# Create GKE cluster with GPU node pool
gcloud container clusters create torchrec-gke \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --num-nodes 1 \
  --enable-ip-alias

# Add GPU node pool with a2-highgpu-8g (8x A100)
gcloud container node-pools create gpu-pool \
  --cluster torchrec-gke \
  --zone us-central1-a \
  --machine-type a2-highgpu-8g \
  --accelerator type=nvidia-tesla-a100,count=8 \
  --num-nodes 2 \
  --min-nodes 1 \
  --max-nodes 4 \
  --enable-autoscaling
```

## Step 2: Install NVIDIA GPU Drivers

```bash
# Install NVIDIA GPU device drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

## Step 3: Configure Filestore (High-Performance Storage)

```bash
# Create Filestore instance (High Scale tier for best performance)
gcloud filestore instances create torchrec-filestore \
  --zone us-central1-a \
  --tier HIGH_SCALE_SSD \
  --file-share name=torchrec_data,capacity=10TB \
  --network name=default
```

## Step 4: Upload Training Data to Cloud Storage

```bash
# Create bucket
gsutil mb -l us-central1 gs://your-criteo-bucket/

# Upload Criteo dataset
gsutil -m cp -r ./criteo_data gs://your-criteo-bucket/criteo/

# Sync to Filestore (from a GCE VM with Filestore mounted)
gsutil -m rsync -r gs://your-criteo-bucket/criteo/ /mnt/filestore/criteo/
```

## Step 5: Run TorchRec Training

### Option A: Using torchrun + Kubernetes Job (Recommended)

`torchrun` is the standard PyTorch distributed launcher, built into PyTorch core.
It provides elastic training, fault tolerance, and works out of the box.

```yaml
# torchrec-training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: torchrec-dlrm-training
spec:
  parallelism: 2
  completions: 2
  completionMode: Indexed
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-a100
      containers:
      - name: torchrec
        image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
        command: ["torchrun"]
        args:
          - "--nnodes=2"
          - "--nproc_per_node=8"
          - "--rdzv_backend=c10d"
          - "--rdzv_endpoint=$(MASTER_ADDR):29500"
          - "train_dlrm.py"
          - "--batch_size=2048"
        resources:
          limits:
            nvidia.com/gpu: 8
        env:
        - name: NCCL_DEBUG
          value: "INFO"
        - name: NCCL_SOCKET_IFNAME
          value: "eth0"
        - name: NCCL_IB_DISABLE
          value: "1"
        volumeMounts:
        - name: filestore-volume
          mountPath: /mnt/filestore
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: filestore-volume
        nfs:
          server: filestore-ip
          path: /torchrec_data
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: "64Gi"
      restartPolicy: Never
```

```bash
kubectl apply -f torchrec-training-job.yaml
```

### Option B: Using Vertex AI

```python
# vertex_ai_training.py
from google.cloud import aiplatform

aiplatform.init(
    project="your-project-id",
    location="us-central1",
)

# Define custom training job
job = aiplatform.CustomJob.from_local_script(
    display_name="torchrec-dlrm-training",
    script_path="train_dlrm.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest",
    requirements=["torchrec", "fbgemm-gpu"],
    replica_count=2,
    machine_type="a2-highgpu-8g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=8,
    args=[
        "--batch_size=2048",
        "--embedding_dim=128",
        "--dataset_path=/gcs/your-criteo-bucket/criteo/",
    ],
)

# Submit training job
job.run(sync=False)
```

### Option C: Using Kubeflow Training Operator (Production)

For production workloads, the [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/trainer/overview/)
is the official PyTorch ecosystem solution for Kubernetes-based distributed training.

```yaml
# torchrec-pytorchjob.yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: torchrec-dlrm-training
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 1
    maxReplicas: 4
  pytorchReplicaSpecs:
    Worker:
      replicas: 2
      template:
        spec:
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-a100
          containers:
          - name: torchrec
            image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
            command:
              - python
              - train_dlrm.py
              - "--batch_size=2048"
            resources:
              limits:
                nvidia.com/gpu: 8
            env:
            - name: NCCL_DEBUG
              value: "INFO"
            - name: NCCL_SOCKET_IFNAME
              value: "eth0"
            - name: NCCL_IB_DISABLE
              value: "1"
            volumeMounts:
            - name: filestore-volume
              mountPath: /mnt/filestore
            - name: dshm
              mountPath: /dev/shm
          volumes:
          - name: filestore-volume
            nfs:
              server: filestore-ip
              path: /torchrec_data
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: "64Gi"
          restartPolicy: OnFailure
```

```bash
# Install Kubeflow Training Operator
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone"

# Submit training job
kubectl apply -f torchrec-pytorchjob.yaml
```

### Option D: Using TorchX (Legacy)

> **Note:** TorchX is no longer actively maintained. The last release was `0.7.0`
> and nightly builds stopped in April 2024. Consider using torchrun + Kubernetes (Option A)
> or Kubeflow Training Operator (Option C) instead.

```bash
pip install torchx[kubernetes]

torchx run \
  -s kubernetes \
  -cfg namespace=default,queue=default \
  dist.ddp \
  -j 2x8 \
  --gpu 8 \
  --cpu 96 \
  --memMB 1300000 \
  --script train_dlrm.py \
  -- \
  --num_embeddings_per_feature "40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36" \
  --embedding_dim 128 \
  --dense_arch_layer_sizes "512,256,128" \
  --over_arch_layer_sizes "1024,1024,512,256,1" \
  --batch_size 2048 \
  --learning_rate 1.0 \
  --dataset_path /mnt/filestore/criteo/
```

## Training Flow Visualization

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          GCP TorchRec Training Flow                                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   1. DATA LOADING (Filestore)                                                           │
│   ═══════════════════════════                                                           │
│                                                                                         │
│   ┌─────────────────┐                                                                   │
│   │ Cloud Storage   │  gsutil rsync                                                     │
│   │  (Criteo 1TB)   │ ─────────────────┐                                                │
│   └─────────────────┘                  │                                                │
│                                        ▼                                                │
│                              ┌─────────────────┐      ┌──────────────────┐              │
│                              │   Filestore     │ ───► │  GPU DataLoader  │              │
│                              │ (High Scale SSD)│      │   (Each Node)    │              │
│                              │  >10 GB/s read  │      └──────────────────┘              │
│                              └─────────────────┘                                        │
│                                                                                         │
│   2. DISTRIBUTED TRAINING (NVLink + VPC)                                                │
│   ══════════════════════════════════════                                                │
│                                                                                         │
│   Node 0 (Rank 0-7)                    Node 1 (Rank 8-15)                               │
│   ┌─────────────────────────┐          ┌─────────────────────────┐                      │
│   │ GPU 0  GPU 1  GPU 2 ... │          │ GPU 0  GPU 1  GPU 2 ... │                      │
│   │ ┌───┐  ┌───┐  ┌───┐     │          │ ┌───┐  ┌───┐  ┌───┐     │                      │
│   │ │ E │  │ E │  │ E │     │          │ │ E │  │ E │  │ E │     │                      │
│   │ │ m │  │ m │  │ m │     │◄────────►│ │ m │  │ m │  │ m │     │                      │
│   │ │ b │  │ b │  │ b │     │  VPC     │ │ b │  │ b │  │ b │     │                      │
│   │ │ e │  │ e │  │ e │     │  100Gbps │ │ e │  │ e │  │ e │     │                      │
│   │ │ d │  │ d │  │ d │     │  NVLink  │ │ d │  │ d │  │ d │     │                      │
│   │ └───┘  └───┘  └───┘     │          │ └───┘  └───┘  └───┘     │                      │
│   │     Embedding Shards    │          │     Embedding Shards    │                      │
│   └─────────────────────────┘          └─────────────────────────┘                      │
│                                                                                         │
│   3. MONITORING & CHECKPOINTING                                                         │
│   ═════════════════════════════                                                         │
│                                                                                         │
│   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐                   │
│   │ Cloud Storage   │     │  Cloud Monitor  │     │   TensorBoard   │                   │
│   │  (Checkpoints)  │     │    Metrics      │     │   (via Vertex)  │                   │
│   │                 │     │  • GPU Util     │     │                 │                   │
│   │  • model.pt     │     │  • Memory       │     │  • Loss curves  │                   │
│   │  • optimizer.pt │     │  • Network I/O  │     │  • Throughput   │                   │
│   └─────────────────┘     └─────────────────┘     └─────────────────┘                   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Cost Optimization

### Use Preemptible/Spot VMs

```bash
# Add preemptible GPU node pool
gcloud container node-pools create gpu-preemptible \
  --cluster torchrec-gke \
  --zone us-central1-a \
  --machine-type a2-highgpu-8g \
  --accelerator type=nvidia-tesla-a100,count=8 \
  --num-nodes 2 \
  --preemptible
```

**Estimated Costs (us-central1):**
| Configuration | On-Demand | Preemptible (70% savings) |
|--------------|-----------|---------------------------|
| 1 node (8 GPU) | ~$29/hr | ~$9/hr |
| 2 nodes (16 GPU) | ~$58/hr | ~$18/hr |
| 4 nodes (32 GPU) | ~$116/hr | ~$36/hr |

### Committed Use Discounts

For long-running training, consider 1-year or 3-year committed use discounts for up to 57% savings.

## Environment Variables

```bash
# Optimal NCCL settings for GCP
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=0

# For multi-node with GPUDirect-TCPX (A3 VMs)
export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX=/run/tcpx
export NCCL_CROSS_NIC=0
export NCCL_ALGO=Ring,Tree
export NCCL_PROTO=Simple
```

## TPU Alternative

GCP also offers TPUs for large-scale training. For TorchRec workloads, consider TPU v4 pods:

```python
# Using TPU v4 with PyTorch/XLA
import torch_xla.core.xla_model as xm

device = xm.xla_device()
model = model.to(device)
```

Note: TPU support for TorchRec is experimental.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU not detected | Verify GPU drivers installed: `nvidia-smi` |
| NCCL timeout | Check firewall rules allow all ports between nodes |
| Filestore slow | Ensure High Scale SSD tier, same zone as compute |
| Preemption | Enable checkpointing, use node anti-affinity |
| Quota exceeded | Request GPU quota increase in IAM console |

## Related Files

- [cloud_component.py](cloud_component.py) - TorchX component for GCP (legacy)
- [gke_cluster.yaml](gke_cluster.yaml) - GKE cluster configuration
- [train_dlrm_gcp.sh](train_dlrm_gcp.sh) - Training launch script

## References

- [GKE GPU Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)
- [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training)
- [Cloud Filestore](https://cloud.google.com/filestore)
- [A2/A3 VM Families](https://cloud.google.com/compute/docs/accelerator-optimized-machines)
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/trainer/overview/)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
