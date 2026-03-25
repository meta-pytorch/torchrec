# TorchRec

**TorchRec** is a PyTorch domain library built to provide common sparsity and parallelism primitives needed for large-scale recommender systems (RecSys). TorchRec allows training and inference of models with large embedding tables sharded across many GPUs and **powers many production RecSys models at Meta**.

## External Presence
TorchRec has been used to accelerate advancements in recommendation systems, some examples:
* [Latest version of Meta's DLRM (Deep Learning Recommendation Model)](https://github.com/facebookresearch/dlrm) is built using TorchRec
* [Disaggregated Multi-Tower: Topology-aware Modeling Technique for Efficient Large-Scale Recommendation](https://arxiv.org/abs/2403.00877) paper
* [The Algorithm ML](https://github.com/twitter/the-algorithm-ml) from Twitter
* [Training Recommendation Models with Databricks](https://docs.databricks.com/en/machine-learning/train-recommender-models.html)
* [Toward 100TB model with Embedding Offloading Paper](https://dl.acm.org/doi/10.1145/3640457.3688037)


## Introduction

To begin learning about TorchRec, check out:
* Our complete [TorchRec Tutorial](https://pytorch.org/tutorials/intermediate/torchrec_intro_tutorial.html)
* The [TorchRec documentation](https://pytorch.org/torchrec/) for an overview of TorchRec and API references


### TorchRec Features
- Parallelism primitives that enable easy authoring of large, performant multi-device/multi-node models using hybrid data-parallelism/model-parallelism.
- Sharders to shard embedding tables with different strategies including data-parallel, table-wise, row-wise, table-wise-row-wise, column-wise, and table-wise-column-wise sharding.
- Planner that can automatically generate optimized sharding plans for models.
- Pipelined training overlapping dataloading device transfer (copy to GPU), inter-device communications (input_dist), and computation (forward, backward) for increased performance.
- Optimized kernels for RecSys powered by [FBGEMM](https://github.com/pytorch/FBGEMM/tree/main).
- Quantization support for reduced precision training and inference, along with optimizing a TorchRec model for C++ inference.
- Common modules for RecSys.
- RecSys datasets (criteo click logs and movielens)
- Examples of end-to-end training such as the DLRM event prediction model trained on criteo click logs dataset.


## Installation

Check out the [Getting Started](https://pytorch.org/torchrec/setup-torchrec.html) section in the documentation for recommended ways to set up Torchrec.

### From Source

**Generally, there isn't a need to build from source**. For most use cases, follow the section above to set up TorchRec. However, to build from source and to get the latest changes, do the following:

1. Install pytorch. See [pytorch documentation](https://pytorch.org/get-started/locally/).
   ```
   CUDA 12.6

   pip install torch --index-url https://download.pytorch.org/whl/nightly/cu126

   CUDA 12.8

   pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128

   CUDA 12.9

   pip install torch --index-url https://download.pytorch.org/whl/nightly/cu129

   CPU

   pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

2. Clone TorchRec.
   ```
   git clone --recursive https://github.com/meta-pytorch/torchrec
   cd torchrec
   ```

3. Install FBGEMM.
   ```
   CUDA 12.6

   pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cu126

   CUDA 12.8

   pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cu128

   CUDA 12.9

   pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cu129

   CPU

   pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

4. Install other requirements.
   ```
   pip install -r requirements.txt
   ```

4. Install TorchRec.
   ```
   python setup.py install develop
   ```

5. Test the installation.
   ```
   GPU mode

   torchx run -s local_cwd dist.ddp -j 1x2 --gpu 2 --script test_installation.py

   CPU Mode

   torchx run -s local_cwd dist.ddp -j 1x2 --script test_installation.py -- --cpu_only
   ```
   See [TorchX](https://pytorch.org/torchx/) for more information on launching distributed and remote jobs.

5. If you want to run a more complex example, please take a look at the torchrec [DLRM example](https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/dlrm_main.py).

## Contributing

See [CONTRIBUTING.md](https://github.com/meta-pytorch/torchrec/blob/main/CONTRIBUTING.md) for details about contributing to TorchRec!

## Citation

If you're using TorchRec, please refer to BibTeX entry to cite this work:
```
@inproceedings{10.1145/3523227.3547387,
author = {Ivchenko, Dmytro and Van Der Staay, Dennis and Taylor, Colin and Liu, Xing and Feng, Will and Kindi, Rahul and Sudarshan, Anirudh and Sefati, Shahin},
title = {TorchRec: a PyTorch Domain Library for Recommendation Systems},
year = {2022},
isbn = {9781450392785},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3523227.3547387},
doi = {10.1145/3523227.3547387},
abstract = {Recommendation Systems (RecSys) comprise a large footprint of production-deployed AI today. The neural network-based recommender systems differ from deep learning models in other domains in using high-cardinality categorical sparse features that require large embedding tables to be trained. In this talk we introduce TorchRec, a PyTorch domain library for Recommendation Systems. This new library provides common sparsity and parallelism primitives, enabling researchers to build state-of-the-art personalization models and deploy them in production. In this talk we cover the building blocks of the TorchRec library including modeling primitives such as embedding bags and jagged tensors, optimized recommender system kernels powered by FBGEMM, a flexible sharder that supports a veriety of strategies for partitioning embedding tables, a planner that automatically generates optimized and performant sharding plans, support for GPU inference and common modeling modules for building recommender system models. TorchRec library is currently used to train large-scale recommender models at Meta. We will present how TorchRec helped Meta’s recommender system platform to transition from CPU asynchronous training to accelerator-based full-sync training.},
booktitle = {Proceedings of the 16th ACM Conference on Recommender Systems},
pages = {482–483},
numpages = {2},
keywords = {information retrieval, recommender systems},
location = {Seattle, WA, USA},
series = {RecSys '22}
}
```

## License
TorchRec is BSD licensed, as found in the [LICENSE](LICENSE) file.
