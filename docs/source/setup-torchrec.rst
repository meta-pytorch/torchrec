.. meta::
   :description: Learn how to set up TorchRec, review system and version compatibility with PyTorch and FBGEMM, install via stable/nightly/source options, and run a simple EmbeddingBagCollection example.
   :keywords: TorchRec, PyTorch, FBGEMM, recommender systems, installation, CUDA, CPU, setup, EmbeddingBagCollection, KeyedJaggedTensor, JaggedTensor


Setup
=====

In this section, we will:

* Understand requirements for using TorchRec
* Set up an environment to integrate TorchRec
* Run basic TorchRec code


System Requirements
-------------------

TorchRec is routinely tested on AWS Linux only and should work in similar environments.
Below demonstrates the compatability matrix that is currently tested:

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - Python Version
     - 3.9, 3.10, 3.11, 3.12
   * - Compute Platform
     - CPU, CUDA 11.8, CUDA 12.1, CUDA 12.4

Aside from those requirements, TorchRec's core dependencies are PyTorch and FBGEMM.
If your system is compatible with both libraries generally, then it should be sufficient for TorchRec.

* `PyTorch requirements <https://pytorch.org/get-started/locally/>`_
* `FBGEMM requirements <https://pytorch.org/FBGEMM/fbgemm-development/BuildInstructions.html>`_


Version Compatability
---------------------

TorchRec and FBGEMM have matching version numbers that are tested together upon release:

* TorchRec 1.0 is compatible with FBGEMM 1.0
* TorchRec 0.8 is compatible with FBGEMM 0.8
* TorchRec 0.8 may not be compatible with FBGEMM 0.7

Furthermore, TorchRec and FBGEMM are released only when a new PyTorch release happens.
Therefore, specific versions of TorchRec and FBGEMM should correspond to a specific PyTorch version:

* TorchRec 1.0 is compatible with PyTorch 2.5
* TorchRec 0.8 is compatible with PyTorch 2.4
* TorchRec 0.8 may not be compatible with PyTorch 2.3

Installation
------------
Below we show installations for CUDA 12.1 as an example. For CPU, CUDA 11.8, or CUDA 12.4, swap ``cu121`` for ``cpu``, ``cu118``, or ``cu124`` respectively.

.. tab-set::

    .. tab-item:: **Stable via pytorch.org**

        .. code-block:: bash

            pip install torch --index-url https://download.pytorch.org/whl/cu121
            pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/cu121
            pip install torchmetrics==1.0.3
            pip install torchrec --index-url https://download.pytorch.org/whl/cu121

    .. tab-item:: **Stable via PyPI (Only for CUDA 12.4)**

        .. code-block:: bash

            pip install torch
            pip install fbgemm-gpu
            pip install torchrec

    .. tab-item:: **Nightly**

        .. code-block:: bash

            pip install torch --index-url https://download.pytorch.org/whl/nightly/cu121
            pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cu121
            pip install torchmetrics==1.0.3
            pip install torchrec --index-url https://download.pytorch.org/whl/nightly/cu121

    .. tab-item:: **Building From Source**

        You also have the ability to build TorchRec from source to develop with the latest
        changes in TorchRec. To build from source, check out this `reference <https://github.com/pytorch/torchrec?tab=readme-ov-file#from-source>`_.


Run a Simple TorchRec Example
------------------------------
Now that we have TorchRec properly set up, let's run some TorchRec code!
Below, we'll run a simple forward pass with TorchRec data types: ``KeyedJaggedTensor`` and ``EmbeddingBagCollection``:

.. code-block:: python

    import torch

    import torchrec
    from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

    ebc = torchrec.EmbeddingBagCollection(
        device="cpu",
        tables=[
            torchrec.EmbeddingBagConfig(
                name="product_table",
                embedding_dim=16,
                num_embeddings=4096,
                feature_names=["product"],
                pooling=torchrec.PoolingType.SUM,
            ),
            torchrec.EmbeddingBagConfig(
                name="user_table",
                embedding_dim=16,
                num_embeddings=4096,
                feature_names=["user"],
                pooling=torchrec.PoolingType.SUM,
            )
        ]
    )

    product_jt = JaggedTensor(
        values=torch.tensor([1, 2, 1, 5]), lengths=torch.tensor([3, 1])
    )
    user_jt = JaggedTensor(values=torch.tensor([2, 3, 4, 1]), lengths=torch.tensor([2, 2]))

    # Q1: How many batches are there, and which values are in the first batch for product_jt and user_jt?
    kjt = KeyedJaggedTensor.from_jt_dict({"product": product_jt, "user": user_jt})

    print("Call EmbeddingBagCollection Forward: ", ebc(kjt))

Save the above code to a file named ``torchrec_example.py``. Then, you should be able to
execute it from your terminal with:

.. code-block:: bash

    python torchrec_example.py

You should see the output ``KeyedTensor`` with the resulting embeddings.
Congrats! You have correctly installed and ran your first TorchRec program!
