Kernel pushed to trec-alitehrani-tpu.
Ensuring jax-tpu-embedding is installed in the pod ...
Running experimental/torch_tpu/scripts/uneven_all2all/repro_tpu_vbe_alltoall.py ...
Source: torch_tpu=aa6805461b6f7216140ea8e4cb5ea6564d389574
TPU topology: 2,2,1,2
World size: 8
Launching: /workspace/torchrec/experimental/torch_tpu/scripts/uneven_all2all/repro_tpu_vbe_alltoall.py
torch=2.15.0.dev20260910+cpu; torch_tpu=0.1.1
world_size=8; equal is the non-VBE control, uneven is the VBE-like collective
rank=0 case=equal start input_splits=[8, 8, 8, 8, 8, 8, 8, 8] output_splits=[8, 8, 8, 8, 8, 8, 8, 8]
rank=0 case=equal collective complete
rank=0 case=equal PASS
rank=0 case=uneven start input_splits=[8, 8, 8, 8, 8, 8, 8, 8] output_splits=[8, 9, 10, 11, 12, 13, 14, 15]
rank=0 case=uneven collective complete
rank=0 case=uneven PASS
