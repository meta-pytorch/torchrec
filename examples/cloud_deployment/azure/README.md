# TorchRec on Microsoft Azure

This guide provides step-by-step instructions for deploying TorchRec distributed training on Microsoft Azure.

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          TorchRec on Azure Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                          Azure Kubernetes Service (AKS)                         │   │
│   │  ┌─────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │                    GPU Node Pool (ND96asr_v4)                           │   │   │
│   │  │                                                                         │   │   │
│   │  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │   │   │
│   │  │   │   Node 0    │    │   Node 1    │    │   Node 2    │   ...          │   │   │
│   │  │   │ 8x A100 GPU │    │ 8x A100 GPU │    │ 8x A100 GPU │                │   │   │
│   │  │   │ 96 vCPUs    │    │ 96 vCPUs    │    │ 96 vCPUs    │                │   │   │
│   │  │   │ 900GB RAM   │    │ 900GB RAM   │    │ 900GB RAM   │                │   │   │
│   │  │   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                │   │   │
│   │  │          │                  │                  │                       │   │   │
│   │  │          └──────────────────┼──────────────────┘                       │   │   │
│   │  │                             │                                          │   │   │
│   │  │                    ┌────────▼────────┐                                 │   │   │
│   │  │                    │   InfiniBand    │   200 Gbps HDR                  │   │   │
│   │  │                    │    Network      │   GPUDirect RDMA                │   │   │
│   │  │                    └─────────────────┘                                 │   │   │
│   │  └─────────────────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                              │
│                    ┌─────────────────────┼─────────────────────┐                        │
│                    │                     │                     │                        │
│                    ▼                     ▼                     ▼                        │
│           ┌───────────────┐     ┌───────────────┐     ┌───────────────┐                 │
│           │  Blob Storage │     │ Azure NetApp  │     │Azure Monitor  │                 │
│           │  (Raw Data)   │     │ Files (NFS)   │     │  (Metrics)    │                 │
│           └───────────────┘     └───────────────┘     └───────────────┘                 │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Recommended VM Sizes

| VM Size | GPUs | GPU Memory | vCPUs | RAM | Network | Use Case |
|---------|------|------------|-------|-----|---------|----------|
| ND96asr_v4 | 8x A100 | 40GB | 96 | 900GB | 200 Gbps IB | Standard training |
| ND96amsr_A100_v4 | 8x A100 | 80GB | 96 | 1.9TB | 200 Gbps IB | Large embeddings |
| ND96isr_H100_v5 | 8x H100 | 80GB | 96 | 1.9TB | 400 Gbps IB | Maximum performance |

## Prerequisites

1. **Azure CLI installed and configured**:
   ```bash
   az login
   az account set --subscription "your-subscription-id"
   ```

2. **kubectl configured**:
   ```bash
   az aks get-credentials --resource-group torchrec-rg --name torchrec-aks
   ```

## Step 1: Create Resource Group

```bash
az group create --name torchrec-rg --location eastus2
```

## Step 2: Create AKS Cluster with GPU Nodes

```bash
# Create AKS cluster
az aks create \
  --resource-group torchrec-rg \
  --name torchrec-aks \
  --node-count 1 \
  --node-vm-size Standard_D4s_v3 \
  --generate-ssh-keys

# Add GPU node pool with ND96asr_v4 (8x A100)
az aks nodepool add \
  --resource-group torchrec-rg \
  --cluster-name torchrec-aks \
  --name gpupool \
  --node-count 2 \
  --node-vm-size Standard_ND96asr_v4 \
  --node-taints sku=gpu:NoSchedule \
  --labels sku=gpu
```

## Step 3: Install NVIDIA Device Plugin

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

## Step 4: Configure Azure NetApp Files (High-Performance Storage)

```bash
# Register NetApp Resource Provider
az provider register --namespace Microsoft.NetApp

# Create NetApp account
az netappfiles account create \
  --resource-group torchrec-rg \
  --name torchrec-netapp \
  --location eastus2

# Create capacity pool (Premium tier for best performance)
az netappfiles pool create \
  --resource-group torchrec-rg \
  --account-name torchrec-netapp \
  --name torchrec-pool \
  --size 4 \
  --service-level Premium

# Create volume
az netappfiles volume create \
  --resource-group torchrec-rg \
  --account-name torchrec-netapp \
  --pool-name torchrec-pool \
  --name torchrec-volume \
  --file-path torchrec \
  --usage-threshold 1024 \
  --vnet your-vnet \
  --subnet your-subnet \
  --protocol-types NFSv3
```

## Step 5: Upload Training Data to Blob Storage

```bash
# Create storage account
az storage account create \
  --name torcrecstorage \
  --resource-group torchrec-rg \
  --location eastus2 \
  --sku Standard_LRS

# Create container
az storage container create --name criteo-data --account-name torcrecstorage

# Upload dataset
az storage blob upload-batch \
  --destination criteo-data \
  --source ./criteo_data \
  --account-name torcrecstorage
```

## Step 6: Run TorchRec Training

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
        sku: gpu
      tolerations:
      - key: "sku"
        operator: "Equal"
        value: "gpu"
        effect: "NoSchedule"
      containers:
      - name: torchrec
        image: mcr.microsoft.com/azureml/openmpi4.1.0-cuda12.1-cudnn8-ubuntu22.04
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
        - name: NCCL_IB_DISABLE
          value: "0"
        - name: NCCL_IB_HCA
          value: "mlx5"
        - name: UCX_TLS
          value: "rc,cuda_copy,cuda_ipc"
        volumeMounts:
        - name: netapp-volume
          mountPath: /mnt/netapp
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: netapp-volume
        nfs:
          server: netapp-server-ip
          path: /torchrec
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: "64Gi"
      restartPolicy: Never
```

```bash
kubectl apply -f torchrec-training-job.yaml
```

### Option B: Using Azure Machine Learning

```python
# azure_ml_training.py
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, AmlCompute
from azure.identity import DefaultAzureCredential

# Connect to workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="torchrec-rg",
    workspace_name="torchrec-workspace",
)

# Define compute cluster
gpu_cluster = AmlCompute(
    name="gpu-cluster",
    type="amlcompute",
    size="Standard_ND96asr_v4",
    min_instances=0,
    max_instances=4,
)
ml_client.compute.begin_create_or_update(gpu_cluster)

# Define training job
job = command(
    code="./training_scripts",
    command="torchrun --nnodes=$AZUREML_NODE_COUNT --nproc_per_node=8 train_dlrm.py --batch_size 2048",
    environment="AzureML-pytorch-2.1-cuda12.1",
    compute="gpu-cluster",
    instance_count=2,
    distribution={"type": "PyTorch"},
)

# Submit job
returned_job = ml_client.jobs.create_or_update(job)
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
            sku: gpu
          tolerations:
          - key: "sku"
            operator: "Equal"
            value: "gpu"
            effect: "NoSchedule"
          containers:
          - name: torchrec
            image: mcr.microsoft.com/azureml/openmpi4.1.0-cuda12.1-cudnn8-ubuntu22.04
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
            - name: NCCL_IB_DISABLE
              value: "0"
            - name: NCCL_IB_HCA
              value: "mlx5"
            volumeMounts:
            - name: netapp-volume
              mountPath: /mnt/netapp
            - name: dshm
              mountPath: /dev/shm
          volumes:
          - name: netapp-volume
            nfs:
              server: netapp-server-ip
              path: /torchrec
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
  --memMB 900000 \
  --script train_dlrm.py \
  -- \
  --num_embeddings_per_feature "40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36" \
  --embedding_dim 128 \
  --dense_arch_layer_sizes "512,256,128" \
  --over_arch_layer_sizes "1024,1024,512,256,1" \
  --batch_size 2048 \
  --learning_rate 1.0 \
  --dataset_path /mnt/netapp/criteo/
```

## Training Flow Visualization

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          Azure TorchRec Training Flow                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   1. DATA LOADING (Azure NetApp Files)                                                  │
│   ════════════════════════════════════                                                  │
│                                                                                         │
│   ┌─────────────────┐                                                                   │
│   │  Blob Storage   │  AzCopy or blobfuse                                               │
│   │  (Criteo 1TB)   │ ─────────────────┐                                                │
│   └─────────────────┘                  │                                                │
│                                        ▼                                                │
│                              ┌─────────────────┐      ┌──────────────────┐              │
│                              │  Azure NetApp   │ ───► │  GPU DataLoader  │              │
│                              │  Files (NFS)    │      │   (Each Node)    │              │
│                              │  4+ GB/s read   │      └──────────────────┘              │
│                              └─────────────────┘                                        │
│                                                                                         │
│   2. DISTRIBUTED TRAINING (InfiniBand)                                                  │
│   ════════════════════════════════════                                                  │
│                                                                                         │
│   Node 0 (Rank 0-7)                    Node 1 (Rank 8-15)                               │
│   ┌─────────────────────────┐          ┌─────────────────────────┐                      │
│   │ GPU 0  GPU 1  GPU 2 ... │          │ GPU 0  GPU 1  GPU 2 ... │                      │
│   │ ┌───┐  ┌───┐  ┌───┐     │          │ ┌───┐  ┌───┐  ┌───┐     │                      │
│   │ │ E │  │ E │  │ E │     │          │ │ E │  │ E │  │ E │     │                      │
│   │ │ m │  │ m │  │ m │     │◄────────►│ │ m │  │ m │  │ m │     │                      │
│   │ │ b │  │ b │  │ b │     │   IB     │ │ b │  │ b │  │ b │     │                      │
│   │ │ e │  │ e │  │ e │     │  HDR     │ │ e │  │ e │  │ e │     │                      │
│   │ │ d │  │ d │  │ d │     │  200Gbps │ │ d │  │ d │  │ d │     │                      │
│   │ └───┘  └───┘  └───┘     │          │ └───┘  └───┘  └───┘     │                      │
│   │     Embedding Shards    │          │     Embedding Shards    │                      │
│   └─────────────────────────┘          └─────────────────────────┘                      │
│                                                                                         │
│   3. MONITORING & CHECKPOINTING                                                         │
│   ═════════════════════════════                                                         │
│                                                                                         │
│   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐                   │
│   │  Blob Storage   │     │  Azure Monitor  │     │   TensorBoard   │                   │
│   │  (Checkpoints)  │     │    Metrics      │     │   (via AML)     │                   │
│   │                 │     │  • GPU Util     │     │                 │                   │
│   │  • model.pt     │     │  • Memory       │     │  • Loss curves  │                   │
│   │  • optimizer.pt │     │  • Network I/O  │     │  • Throughput   │                   │
│   └─────────────────┘     └─────────────────┘     └─────────────────┘                   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Cost Optimization

### Use Spot VMs

```bash
# Add spot node pool
az aks nodepool add \
  --resource-group torchrec-rg \
  --cluster-name torchrec-aks \
  --name gpuspot \
  --node-count 2 \
  --node-vm-size Standard_ND96asr_v4 \
  --priority Spot \
  --eviction-policy Delete \
  --spot-max-price -1
```

**Estimated Costs (East US 2):**
| Configuration | On-Demand | Spot (65% savings) |
|--------------|-----------|-------------------|
| 1 node (8 GPU) | ~$27/hr | ~$9.50/hr |
| 2 nodes (16 GPU) | ~$54/hr | ~$19/hr |
| 4 nodes (32 GPU) | ~$108/hr | ~$38/hr |

### Reserved Instances

For long-running training, consider 1-year or 3-year reserved instances for up to 60% savings.

## Environment Variables

```bash
# Optimal NCCL settings for Azure InfiniBand
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=eth0
export UCX_TLS=rc,cuda_copy,cuda_ipc
export UCX_NET_DEVICES=mlx5_ib0:1
export NCCL_IB_TIMEOUT=22
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| InfiniBand not detected | Verify ND-series VM, check `ibstat` |
| NCCL timeout | Increase `NCCL_IB_TIMEOUT`, verify NSG rules |
| NetApp slow | Ensure Premium tier, same region as compute |
| Node eviction (Spot) | Enable checkpointing, use tolerations |

## Related Files

- [cloud_component.py](cloud_component.py) - TorchX component for Azure (legacy)
- [aks_cluster.yaml](aks_cluster.yaml) - AKS cluster configuration
- [train_dlrm_azure.sh](train_dlrm_azure.sh) - Training launch script

## References

- [Azure ND-series VMs](https://docs.microsoft.com/azure/virtual-machines/nd-series)
- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/)
- [Azure NetApp Files](https://docs.microsoft.com/azure/azure-netapp-files/)
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/trainer/overview/)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
