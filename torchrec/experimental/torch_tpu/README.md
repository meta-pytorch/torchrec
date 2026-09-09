# PyTorch TPU SparseCore Embedding (`torchrec.experimental.torch_tpu`)

`torchrec.experimental.torch_tpu` provides Google TPU **SparseCore** hardware-accelerated replacements for PyTorch **TorchRec** recommendation model embedding modules.

---

## Overview & Why TPU SparseCore?

In standard PyTorch and **TorchRec**, recommendation models handle high-cardinality categorical features using tables wrapped in an `EmbeddingBagCollection` (EBC) or `EmbeddingCollection` (EC). On GPUs and CPUs, looking up and optimizing massive sparse tables requires complex software sharding (`DistributedModelParallel`), host-to-device memory traffic, and separate optimizer kernels (`KeyedOptimizer`).

Google TPUs feature specialized hardware accelerators called **SparseCore (SC)**, designed explicitly to execute sparse embedding lookups, communication, and gradient updates at high throughput.

The `torchrec.experimental.torch_tpu` library provides drop-in, hardware-accelerated replacements for core TorchRec modules:
* Executes both forward sparse lookups and backward optimizer updates natively in **SparseCore** hardware.
* Eliminates autograd `.grad` tensor overhead on HBM by fusing optimizer step updates into the backward pass.
* Offloads CSR-to-COO layout conversion and partition padding to a multi-threaded CPU C++ PyBind backend (`SparseCoreInputPreprocessor`).

---

## Architectural Comparison: TorchRec vs. `torchrec.experimental.torch_tpu`

| Component | Standard TorchRec | TPU `torchrec.experimental.torch_tpu` | Key Difference / Benefit |
| :--- | :--- | :--- | :--- |
| **Table Configuration** | `torchrec.EmbeddingBagConfig` <br> `torchrec.EmbeddingConfig` | `SparseCoreEmbeddingConfig(config=..., max_ids_per_partition=256, ...)` | Wraps standard TorchRec configs with TPU SparseCore partition capacity and sequence bounds. |
| **Pooled Collection (EBC)** | `torchrec.EmbeddingBagCollection` | `SparseCoreFusedEmbeddingBagCollection` | Replaces CPU/GPU EBC. Automatically handles MOD sharding across SparseCore partitions and chips. |
| **Unpooled Collection (EC)** | `torchrec.EmbeddingCollection` | `SparseCoreFusedEmbeddingCollection` | Replaces EC for sequence models (e.g., HSTU, Shakespeare). Requires `max_seq_len` in config. |
| **Optimizer & Backward Pass** | Separate `KeyedOptimizer` / `apply_optimizer_in_backward` | **Fused into the embedding module** via `optimizer_type` & `optimizer_kwargs` | Gradient computation and weight/momentum updates happen in-place on SparseCore during `loss.backward()`. |
| **Input Preprocessing** | Raw `KeyedJaggedTensor` (KJT) passed directly to EBC/EC | CPU `SparseCoreInputPreprocessor` $\rightarrow$ `KeyedSparseCorePreprocessedInput` | Transforms KJT into padded COO/T8 tensors on CPU before `.to("tpu")` device transfer. |
| **Distributed Sharding** | Explicit `DistributedModelParallel` (DMP) + `EmbeddingShardingPlanner` | **Automatic sharding** across `global_device_count` $\times$ `num_sc_per_device` | No DMP wrapper needed around embedding modules. |

---

## Step-by-Step Migration Guide

### Step 1: Wrap Table Configurations with `SparseCoreEmbeddingConfig`

In standard TorchRec, you define tables using `EmbeddingBagConfig` (for pooled embeddings) or `EmbeddingConfig` (for unpooled sequence embeddings).

When migrating to `torchrec.experimental.torch_tpu`, wrap your existing configs in `SparseCoreEmbeddingConfig` to specify hardware partition capacity:

```python
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.experimental.torch_tpu.modules.embedding_configs import SparseCoreEmbeddingConfig

# 1. Define standard TorchRec config
base_config = EmbeddingBagConfig(
    name="item_table",
    embedding_dim=64,       # Must be divisible by 8
    num_embeddings=4096,    # Will be rounded up to multiple of (num_sc * 8)
    feature_names=["item_id"],
    pooling=PoolingType.SUM,
)

# 2. Wrap for TPU SparseCore
sc_table_config = SparseCoreEmbeddingConfig(
    config=base_config,
    max_ids_per_partition=256,
    max_unique_ids_per_partition=256,
    suggested_coo_buffer_size_per_device=32,
)
```

> **Note:** For unpooled sequence tables (`EmbeddingConfig`), you **must** provide `max_seq_len` to `SparseCoreEmbeddingConfig(..., max_seq_len=64)` so the TPU compiler can statically allocate padded sequence buffers.

---

### Step 2: Replace Embedding Collections with Fused SparseCore Collections

Replace `EmbeddingBagCollection` with `SparseCoreFusedEmbeddingBagCollection` (or `SparseCoreFusedEmbeddingCollection` for unpooled embeddings).

Because TPU SparseCore fuses embedding lookups and backward optimizer updates into a single kernel (`SparseDenseMatmul`), pass your optimizer configuration directly into the module constructor:

```python
import torch
from torchrec.experimental.torch_tpu.modules.fused_embedding_modules import SparseCoreFusedEmbeddingBagCollection

embedding_layer = SparseCoreFusedEmbeddingBagCollection(
    tables=[sc_table_config],
    optimizer_type=torch.optim.SGD,     # Supported: torch.optim.SGD, Adagrad, Adam
    optimizer_kwargs={"lr": 0.05},      # Learning rate & optimizer hyperparameters
    batch_size=local_batch_size,        # Process-local batch size
    global_device_count=world_size,     # Total TPU chips across hosts
    num_sc_per_device=2,                # SparseCore virtual partitions per TPU chip
)
```

#### Supported Optimizer Hyperparameters (`optimizer_kwargs`)
`torchrec.experimental.torch_tpu` supports standard PyTorch optimizer parameter names:
* **SGD**: `{"lr": 0.05}`
* **Adagrad**: `{"lr": 0.1, "initial_accumulator_value": 0.1, "eps": 1e-10}`
* **Adam**: `{"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8}`

*(Note: Legacy argument names `learning_rate`, `beta1`, `beta2`, and `epsilon` are also seamlessly accepted).*

#### Modifying Learning Rate During Training
To schedule or update the learning rate of the embedding tables across epochs, call:
```python
embedding_layer.set_learning_rate(new_lr)
```

---

### Step 3: Setup CPU Input Preprocessing & Dataloading

TPU SparseCore operates on structured CSR/COO layouts (`KeyedSparseCorePreprocessedInput`). Use `SparseCoreInputPreprocessor` to convert raw `KeyedJaggedTensor` (KJT) feature inputs on the CPU before copying to TPU.

```python
from torchrec.experimental.torch_tpu.datasets.input_preprocessing import SparseCoreInputPreprocessor
from torchrec.experimental.torch_tpu.datasets.dataloader import SparseCoreBatch

# 1. Instantiate CPU preprocessor once before the training loop
preprocessor = SparseCoreInputPreprocessor(
    tables=[sc_table_config],
    batch_size=local_batch_size,
    global_device_count=world_size,
    num_sc_per_device=2,
    allow_id_dropping=False,
)

# 2. Inside your dataloader / training loop:
# Convert raw TorchRec Batch (CPU) into a TPU SparseCore batch
tpu_batch = SparseCoreBatch.from_batch(raw_batch, preprocessor).to("tpu")
```

---

### Step 4: Update Model Forward & Backward Loop

The forward call syntax for `SparseCoreFusedEmbeddingBagCollection` is identical to TorchRec: it receives `tpu_batch.sparse_features` and returns a standard `torchrec.KeyedTensor`:

```python
# Forward pass
pooled_embeddings = embedding_layer(tpu_batch.sparse_features)
# pooled_embeddings["item_id"] -> Tensor of shape (batch_size, embedding_dim)

# Combine with dense features and compute loss
loss = compute_loss(pooled_embeddings, dense_features, labels)

# Backward pass
loss.backward()  # <--- Automatically runs TPU SparseCore backward & optimizer step!
```

> **Tip:** When defining your dense model optimizer (e.g., for MLPs and interaction layers), filter out embedding parameters so they are not doubly optimized by PyTorch autograd:
> ```python
> dense_params = [
>     p for name, p in model.named_parameters() if "embedding_layer" not in name
> ]
> dense_optimizer = torch.optim.SGD(dense_params, lr=0.01)
>
> # Training step:
> dense_optimizer.zero_grad()
> loss.backward()
> dense_optimizer.step()  # Only updates dense MLP parameters
> ```

---

## Complete Code Comparison: DLRM Training (Before vs. After)

### Before: Standard TorchRec (`DLRM`)

```python
import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.datasets.utils import Batch

# 1. Table Configs & EmbeddingBagCollection
eb_config = EmbeddingBagConfig(
    name="t_user",
    embedding_dim=64,
    num_embeddings=4096,
    feature_names=["f_user"],
    pooling=PoolingType.SUM,
)
ebc = EmbeddingBagCollection(tables=[eb_config])

# 2. Model & Optimizer
dlrm_model = DLRM(
    embedding_bag_collection=ebc,
    dense_in_features=100,
    dense_arch_layer_sizes=[64, 64],
    over_arch_layer_sizes=[32, 1],
).to("cuda")

train_model = DLRMTrain(dlrm_model)
optimizer = torch.optim.SGD(train_model.parameters(), lr=0.1)

# 3. Training Step (Raw KJT directly on GPU)
optimizer.zero_grad()
loss, _ = train_model(batch.to("cuda"))
loss.backward()
optimizer.step()  # Updates both dense and embedding weights
```

---

### After: Migrated TPU SparseCore (`DLRM` with `torchrec.experimental.torch_tpu`)

```python
import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.experimental.torch_tpu.modules.embedding_configs import SparseCoreEmbeddingConfig
from torchrec.experimental.torch_tpu.modules.fused_embedding_modules import SparseCoreFusedEmbeddingBagCollection
from torchrec.experimental.torch_tpu.datasets.input_preprocessing import SparseCoreInputPreprocessor
from torchrec.experimental.torch_tpu.datasets.dataloader import SparseCoreBatch
from torchrec.models.dlrm import DLRM, DLRMTrain

device = torch.device("tpu")
batch_size = 128

# 1. Wrap Table Config in SparseCoreEmbeddingConfig
sc_config = SparseCoreEmbeddingConfig(
    config=EmbeddingBagConfig(
        name="t_user",
        embedding_dim=64,
        num_embeddings=4096,
        feature_names=["f_user"],
        pooling=PoolingType.SUM,
    ),
    max_ids_per_partition=256,
    max_unique_ids_per_partition=256,
)

# 2. Fused TPU SparseCore EmbeddingBagCollection
ebc = SparseCoreFusedEmbeddingBagCollection(
    tables=[sc_config],
    optimizer_type=torch.optim.SGD,      # Fused optimizer inside SparseCore
    optimizer_kwargs={"lr": 0.1},
    batch_size=batch_size,
    global_device_count=1,
    num_sc_per_device=2,
)

dlrm_model = DLRM(
    embedding_bag_collection=ebc,
    dense_in_features=100,
    dense_arch_layer_sizes=[64, 64],
    over_arch_layer_sizes=[32, 1],
).to(device)

train_model = DLRMTrain(dlrm_model)

# 3. Optimize ONLY dense parameters on host/TPU
dense_params = [
    p for name, p in train_model.named_parameters() if "embedding_bags" not in name
]
dense_optimizer = torch.optim.SGD(dense_params, lr=0.1)

# 4. CPU Preprocessor Setup
preprocessor = SparseCoreInputPreprocessor(
    tables=[sc_config],
    batch_size=batch_size,
)

# 5. Training Step
dense_optimizer.zero_grad()

# Preprocess on CPU, then copy to TPU
tpu_batch = SparseCoreBatch.from_batch(raw_batch, preprocessor).to(device)

loss, _ = train_model(tpu_batch)
loss.backward()          # TPU SparseCore updates embedding weights in hardware
dense_optimizer.step()   # Host updates dense MLP parameters
```

---

## Migrating Unpooled Sequence Models (`DLRM-HSTU`, `Shakespeare`)

For models that require unpooled sequence embeddings (where each token or categorical ID retains its embedding vector across `max_seq_len` time steps), use **`SparseCoreFusedEmbeddingCollection`**.

### Config & Layer Setup
```python
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.experimental.torch_tpu.modules.embedding_configs import SparseCoreEmbeddingConfig
from torchrec.experimental.torch_tpu.modules.fused_embedding_modules import SparseCoreFusedEmbeddingCollection

# 1. Define unpooled EmbeddingConfig and specify max_seq_len
seq_config = SparseCoreEmbeddingConfig(
    config=EmbeddingConfig(
        name="token_table",
        embedding_dim=128,
        num_embeddings=50000,
        feature_names=["token_ids"],
    ),
    max_seq_len=64,          # REQUIRED for unpooled sequence lookups
    max_ids_per_partition=512,
    max_unique_ids_per_partition=512,
)

# 2. Create unpooled collection
embedding_collection = SparseCoreFusedEmbeddingCollection(
    tables=[seq_config],
    optimizer_type=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8},
    batch_size=batch_size,
    global_device_count=world_size,
    num_sc_per_device=2,
)
```

### Forward Output Format
Unlike `EmbeddingBagCollection` (which returns a `KeyedTensor`), calling `SparseCoreFusedEmbeddingCollection` returns a dictionary of `JaggedTensor` objects keyed by feature name:

```python
# Forward call returns Dict[str, JaggedTensor]
output_dict = embedding_collection(tpu_batch.sparse_features)

token_embeddings = output_dict["token_ids"].values()
# token_embeddings shape -> (actual_num_ids, embedding_dim)
```

---

## Distributed Checkpointing (DCP) & CPU Inference Export

`torchrec.experimental.torch_tpu` integrates natively with PyTorch Distributed Checkpoint (`torch.distributed.checkpoint` / `dcp`) via custom planners: **`SparseCoreSavePlanner`** and **`SparseCoreLoadPlanner`**.

Because embedding tables are MOD-sharded across multiple TPU SparseCores globally, DCP allows:
1. **Zero-Copy Distributed Save**: Each TPU rank streams its local shard to storage with global chunk metadata.
2. **TPU Training Resumption**: Each TPU rank reloads its corresponding shard for seamless multi-device training resumption.
3. **Cross-Topology Loading**: Allows loading checkpoints onto topologies with different number of ranks or SparseCores, enabling flexible re-sharding.
4. **CPU Inference Unsharding**: Loads multi-device checkpoints directly into standard CPU TorchRec `EmbeddingBagCollection` / `EmbeddingCollection` models with vectorized in-memory table reconstruction.

### 1. Saving Distributed Checkpoint on TPU

```python
import torch.distributed.checkpoint as dcp
from torchrec.experimental.torch_tpu.checkpoint.planners import SparseCoreSavePlanner

# Save state dict containing TPU SparseCore embedding tables
# No need to isolate embedding tables! Pass full state_dict.
save_state_dict = dlrm_model.state_dict()

dcp.save(
    state_dict=save_state_dict,
    storage_writer=dcp.FileSystemWriter("/tmp/my_checkpoint_dir"),
    planner=SparseCoreSavePlanner(num_sc_per_device=2),
)
```

---

### 2. Loading for Multi-Device TPU Training Resumption

```python
from torchrec.experimental.torch_tpu.checkpoint.planners import SparseCoreLoadPlanner

# Fresh model instance on TPU
load_state_dict = new_dlrm_model.state_dict()

dcp.load(
    state_dict=load_state_dict,
    storage_reader=dcp.FileSystemReader("/tmp/my_checkpoint_dir"),
    planner=SparseCoreLoadPlanner(
        num_sc_per_device=2,
        unshard_for_cpu=False,
    ),
)
```

---

### 3. Loading for CPU Inference (Unsharding to Standard TorchRec)

When serving models on CPU or GPUs, load the checkpoint directly into standard TorchRec `EmbeddingBagCollection` or `EmbeddingCollection`:

```python
import torch
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.experimental.torch_tpu.checkpoint.planners import SparseCoreLoadPlanner

# Standard CPU TorchRec module
cpu_ebc = EmbeddingBagCollection(tables=[base_config], device=torch.device("cpu"))
cpu_state_dict = cpu_ebc.state_dict()

# Automatically unshard MOD-sharded tables in-place during load
dcp.load(
    state_dict=cpu_state_dict,
    storage_reader=dcp.FileSystemReader("/tmp/my_checkpoint_dir"),
    planner=SparseCoreLoadPlanner(
        num_sc_per_device=2,
        unshard_for_cpu=True,
    ),
)
```

---
