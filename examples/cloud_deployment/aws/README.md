# TorchRec on AWS

This guide provides step-by-step instructions for deploying TorchRec distributed training on Amazon Web Services (AWS).

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            TorchRec on AWS Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              Amazon EKS Cluster                                 │   │
│   │  ┌─────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │                    GPU Node Group (p4d.24xlarge)                        │   │   │
│   │  │                                                                         │   │   │
│   │  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │   │   │
│   │  │   │   Node 0    │    │   Node 1    │    │   Node 2    │   ...          │   │   │
│   │  │   │ 8x A100 GPU │    │ 8x A100 GPU │    │ 8x A100 GPU │                │   │   │
│   │  │   │ 96 vCPUs    │    │ 96 vCPUs    │    │ 96 vCPUs    │                │   │   │
│   │  │   │ 1.1TB RAM   │    │ 1.1TB RAM   │    │ 1.1TB RAM   │                │   │   │
│   │  │   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                │   │   │
│   │  │          │                  │                  │                       │   │   │
│   │  │          └──────────────────┼──────────────────┘                       │   │   │
│   │  │                             │                                          │   │   │
│   │  │                    ┌────────▼────────┐                                 │   │   │
│   │  │                    │   EFA Network   │   400 Gbps per node             │   │   │
│   │  │                    │  (Low Latency)  │   GPUDirect RDMA                │   │   │
│   │  │                    └─────────────────┘                                 │   │   │
│   │  └─────────────────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                              │
│                    ┌─────────────────────┼─────────────────────┐                        │
│                    │                     │                     │                        │
│                    ▼                     ▼                     ▼                        │
│           ┌───────────────┐     ┌───────────────┐     ┌───────────────┐                 │
│           │  Amazon S3    │     │ FSx Lustre    │     │  CloudWatch   │                 │
│           │  (Raw Data)   │     │ (Fast I/O)    │     │  (Monitoring) │                 │
│           └───────────────┘     └───────────────┘     └───────────────┘                 │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Recommended Instance Types

| Instance | GPUs | GPU Memory | vCPUs | RAM | Network | Use Case |
|----------|------|------------|-------|-----|---------|----------|
| p4d.24xlarge | 8x A100 | 40GB | 96 | 1.1TB | 400 Gbps EFA | Standard training |
| p4de.24xlarge | 8x A100 | 80GB | 96 | 1.1TB | 400 Gbps EFA | Large embeddings |
| p5.48xlarge | 8x H100 | 80GB | 192 | 2TB | 3200 Gbps EFA | Maximum performance |

## Prerequisites

1. **AWS CLI configured**:
   ```bash
   aws configure
   ```

2. **eksctl installed**:
   ```bash
   curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
   sudo mv /tmp/eksctl /usr/local/bin
   ```

3. **kubectl configured**:
   ```bash
   aws eks update-kubeconfig --name torchrec-cluster --region us-west-2
   ```

## Step 1: Create EKS Cluster

```bash
# Create EKS cluster with GPU node group
eksctl create cluster -f eks_cluster.yaml
```

Or manually:

```bash
eksctl create cluster \
  --name torchrec-cluster \
  --region us-west-2 \
  --version 1.28 \
  --nodegroup-name gpu-nodes \
  --node-type p4d.24xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed
```

## Step 2: Install NVIDIA Device Plugin

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

## Step 3: Configure FSx for Lustre (High-Performance Storage)

```bash
# Create FSx for Lustre file system linked to S3
aws fsx create-file-system \
  --file-system-type LUSTRE \
  --storage-capacity 1200 \
  --subnet-ids subnet-xxxxxxxx \
  --lustre-configuration ImportPath=s3://your-criteo-bucket/,DeploymentType=PERSISTENT_1
```

## Step 4: Upload Training Data to S3

```bash
# Upload Criteo dataset to S3
aws s3 sync ./criteo_data s3://your-criteo-bucket/criteo/
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
      subdomain: torchrec-headless
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
        - name: NCCL_IB_DISABLE
          value: "0"
        - name: FI_EFA_USE_DEVICE_RDMA
          value: "1"
        - name: FI_PROVIDER
          value: "efa"
        - name: NCCL_NET_GDR_LEVEL
          value: "2"
        volumeMounts:
        - name: fsx-volume
          mountPath: /fsx
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: fsx-volume
        persistentVolumeClaim:
          claimName: fsx-pvc
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: "64Gi"
      restartPolicy: Never
```

```bash
kubectl apply -f torchrec-training-job.yaml
```

### Option B: Using Kubeflow Training Operator (Production)

For production workloads, the [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/trainer/overview/)
is the official PyTorch ecosystem solution for Kubernetes-based distributed training.
It manages `PyTorchJob` resources with built-in support for elastic training, gang scheduling,
and automatic `torchrun` configuration.

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
            - name: FI_EFA_USE_DEVICE_RDMA
              value: "1"
            - name: FI_PROVIDER
              value: "efa"
            volumeMounts:
            - name: fsx-volume
              mountPath: /fsx
            - name: dshm
              mountPath: /dev/shm
          volumes:
          - name: fsx-volume
            persistentVolumeClaim:
              claimName: fsx-pvc
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

### Option C: Using TorchX (Legacy)

> **Note:** TorchX is no longer actively maintained. The last release was `0.7.0`
> and nightly builds stopped in April 2024. Consider using torchrun + Kubernetes (Option A)
> or Kubeflow Training Operator (Option B) instead.

```bash
# Install TorchX
pip install torchx[kubernetes]

# Run distributed training
torchx run \
  -s kubernetes \
  -cfg namespace=default,queue=default \
  dist.ddp \
  -j 2x8 \
  --gpu 8 \
  --cpu 96 \
  --memMB 1100000 \
  --script train_dlrm.py \
  -- \
  --num_embeddings_per_feature "40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36" \
  --embedding_dim 128 \
  --dense_arch_layer_sizes "512,256,128" \
  --over_arch_layer_sizes "1024,1024,512,256,1" \
  --batch_size 2048 \
  --learning_rate 1.0 \
  --dataset_path /fsx/criteo/
```

## Training Flow Visualization

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           AWS TorchRec Training Flow                                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   1. DATA LOADING (FSx for Lustre)                                                      │
│   ════════════════════════════════                                                      │
│                                                                                         │
│   ┌─────────────────┐                                                                   │
│   │    Amazon S3    │  Automatic sync                                                   │
│   │  (Criteo 1TB)   │ ─────────────────┐                                                │
│   └─────────────────┘                  │                                                │
│                                        ▼                                                │
│                              ┌─────────────────┐      ┌──────────────────┐              │
│                              │  FSx for Lustre │ ───► │  GPU DataLoader  │              │
│                              │  (1.2 TB cache) │      │   (Each Node)    │              │
│                              │  >10 GB/s read  │      └──────────────────┘              │
│                              └─────────────────┘                                        │
│                                                                                         │
│   2. DISTRIBUTED TRAINING (EFA Network)                                                 │
│   ═════════════════════════════════════                                                 │
│                                                                                         │
│   Node 0 (Rank 0-7)                    Node 1 (Rank 8-15)                               │
│   ┌─────────────────────────┐          ┌─────────────────────────┐                      │
│   │ GPU 0  GPU 1  GPU 2 ... │          │ GPU 0  GPU 1  GPU 2 ... │                      │
│   │ ┌───┐  ┌───┐  ┌───┐     │          │ ┌───┐  ┌───┐  ┌───┐     │                      │
│   │ │ E │  │ E │  │ E │     │          │ │ E │  │ E │  │ E │     │                      │
│   │ │ m │  │ m │  │ m │     │◄────────►│ │ m │  │ m │  │ m │     │                      │
│   │ │ b │  │ b │  │ b │     │   EFA    │ │ b │  │ b │  │ b │     │                      │
│   │ │ e │  │ e │  │ e │     │  400Gbps │ │ e │  │ e │  │ e │     │                      │
│   │ │ d │  │ d │  │ d │     │ GPUDirect│ │ d │  │ d │  │ d │     │                      │
│   │ └───┘  └───┘  └───┘     │          │ └───┘  └───┘  └───┘     │                      │
│   │     Embedding Shards    │          │     Embedding Shards    │                      │
│   └─────────────────────────┘          └─────────────────────────┘                      │
│                                                                                         │
│   3. CHECKPOINTING & MONITORING                                                         │
│   ═════════════════════════════                                                         │
│                                                                                         │
│   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐                   │
│   │   Amazon S3     │     │   CloudWatch    │     │   TensorBoard   │                   │
│   │  (Checkpoints)  │     │    Metrics      │     │   (via S3)      │                   │
│   │                 │     │  • GPU Util     │     │                 │                   │
│   │  • model.pt     │     │  • Memory       │     │  • Loss curves  │                   │
│   │  • optimizer.pt │     │  • Network I/O  │     │  • Throughput   │                   │
│   └─────────────────┘     └─────────────────┘     └─────────────────┘                   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Cost Optimization

### Use Spot Instances

```yaml
# In eksctl config, add spot instances
managedNodeGroups:
  - name: gpu-spot-nodes
    instanceTypes: ["p4d.24xlarge"]
    spot: true
    minSize: 0
    maxSize: 4
```

**Estimated Costs (us-west-2):**
| Configuration | On-Demand | Spot (70% savings) |
|--------------|-----------|-------------------|
| 1 node (8 GPU) | ~$32/hr | ~$10/hr |
| 2 nodes (16 GPU) | ~$64/hr | ~$20/hr |
| 4 nodes (32 GPU) | ~$128/hr | ~$40/hr |

### Enable Checkpointing for Spot

```python
# In your training script
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

# Save every 1000 iterations for spot instance recovery
if iteration % 1000 == 0:
    save_checkpoint(model, optimizer, epoch, f"s3://bucket/checkpoint_{iteration}.pt")
```

## Environment Variables

```bash
# Optimal NCCL settings for AWS EFA
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export NCCL_PROTO=simple
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| EFA not detected | Ensure EFA driver is installed: `modinfo efa` |
| NCCL timeout | Check security groups allow all traffic between nodes |
| FSx slow | Verify FSx is in same AZ as compute nodes |
| OOM errors | Reduce batch size or use p4de (80GB) instances |

## Related Files

- [cloud_component.py](cloud_component.py) - TorchX component for AWS (legacy)
- [eks_cluster.yaml](eks_cluster.yaml) - EKS cluster configuration
- [train_dlrm_aws.sh](train_dlrm_aws.sh) - Training launch script

## References

- [AWS EKS GPU Documentation](https://docs.aws.amazon.com/eks/latest/userguide/gpu-ami.html)
- [FSx for Lustre](https://aws.amazon.com/fsx/lustre/)
- [AWS EFA](https://aws.amazon.com/hpc/efa/)
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/trainer/overview/)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
