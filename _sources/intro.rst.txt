.. meta::
    :description: TorchRec is a PyTorch domain library for scalable recommendation systems with embeddings, sharding, distributed training, and production inference.
    :keywords: TorchRec, PyTorch, recommendation systems, embeddings, sharding, distributed training, jagged tensors, FSDP, inference, RecSys

Intro
=====

TorchRec is a PyTorch domain library for building and scaling recommendation systems. It provides composable building blocks for industry-scale RecSys workloads, with a focus on sparse features, embedding tables, sharding, distributed training, and inference.

Key capabilities of TorchRec include:
- **Embedding modules:** High-performance embedding bags and tables optimized for sparse categorical features.
- **Sharding and parallelism:** Built-in support for model, table, and row-wise sharding across GPUs and nodes.
- **Distributed training:** Integrations with PyTorch Distributed and FSDP to train massive models efficiently.
- **Feature processing:** Utilities for jagged tensors, pooling, and common RecSys data structures.


.. toctree::
   :maxdepth: 2

   overview.rst
   high-level-arch.rst
   concepts.rst
