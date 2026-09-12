Source: torch_tpu=aa6805461b6f7216140ea8e4cb5ea6564d389574
Multi-host launch: node_rank=0/16 master=gem-alitehrani-tpu-64chip-worker-0-0.gem-alitehrani-tpu-64chip:29500
  topology=4,4,4,2  nproc_per_node=8  (world=128)
  launching /workspace/torchrec/experimental/torch_tpu/scripts/uneven_all2all/repro_tpu_vbe_alltoall.py 
  stack rlimit: 8388608 KB
torch=2.15.0.dev20260910+cpu; torch_tpu=0.1.1
world_size=128; equal is the non-VBE control, uneven is the VBE-like collective
rank=0 case=equal start input_splits=[8, 8, 8, 8, 8, 8, 8, 8] output_splits=[8, 8, 8, 8, 8, 8, 8, 8]
rank=0 case=equal collective complete
rank=0 case=equal PASS
rank=0 case=uneven start input_splits=[8, 8, 8, 8, 8, 8, 8, 8] output_splits=[8, 9, 10, 11, 12, 13, 14, 15]
rank=0 case=uneven collective complete
[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/torchrec/experimental/torch_tpu/scripts/uneven_all2all/repro_tpu_vbe_alltoall.py", line 94, in <module>
[rank0]:     main()
[rank0]:   File "/workspace/torchrec/experimental/torch_tpu/scripts/uneven_all2all/repro_tpu_vbe_alltoall.py", line 89, in main
[rank0]:     _run_case(case)
[rank0]:   File "/workspace/torchrec/experimental/torch_tpu/scripts/uneven_all2all/repro_tpu_vbe_alltoall.py", line 64, in _run_case
[rank0]:     torch.testing.assert_close(actual, expected)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/torch/testing/_comparison.py", line 1689, in assert_close
[rank0]:     raise error_metas[0].to_error(msg)
[rank0]: AssertionError: Tensor-likes are not equal!
[rank0]: Mismatched elements: 1389 / 1472 (94.4%)
[rank0]: Greatest absolute difference: 2146483647 at index (8,)
[rank0]: Greatest relative difference: 2146.483642578125 at index (8,)
[rank0]:[W910 18:38:24.800725968 env_vars.h:296] Warning: the TORCH_SHOW_CPP_STACKTRACES environment variable is an experimental feature and may change or be removed without notice. (function operator())
