# TorchRec Benchmark
## benchmark_train_pipeline usage
- internal:
```
buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_train_pipeline -- \
    --yaml_config=fbcode/torchrec/distributed/benchmark/yaml/sparse_data_dist_base.yml \
    --name=sparse_data_dist_base_$(hg whereami | cut -c 1-10 || echo $USER) # overrides the yaml config
```
- oss:
```
python -m torchrec.distributed.benchmark.benchmark_train_pipeline \
    --yaml_config=fbcode/torchrec/distributed/benchmark/yaml/sparse_data_dist_base.yml \
    --name=sparse_data_dist_base_$(git rev-parse --short HEAD || echo $USER) # overrides the yaml config
```

## benchmark_comms usage
- internal:
```
buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_comms -- \
    a2a_single --name=a2a_sync_base-$(hg whereami | cut -c 1-10) --memory_snapshot=true
```
- oss:
```
python -m torchrec.distributed.benchmark.benchmark_comms \
  a2a_single --name=a2a_sync_base-$(git rev-parse --short HEAD || echo $USER) --memory_snapshot=true
```

## benchmark_primitive / benchmark_ebc usage
Collective (`--benchmark=primitive`) and sharded-EBC (`--benchmark=module`) micro
benchmarks, dispatched by `benchmark_launcher`.
- local (GPU, or CPU with gloo):
```
python -m torchrec.distributed.test_utils.benchmark_utils.benchmark_launcher \
    --mode=local --benchmark=primitive --name=kt_a2a --world-size=2
```
- TPU (one rank per core, `torch_tpu` installed, topology env exported by its launcher;
  `module` supports only `column_wise` without the pipeline):
```
eval "$(python -m torch.tpu.distributed)"
python -m torch.distributed.run --nproc_per_node="${WORLD_SIZE}" \
    -m torchrec.distributed.test_utils.benchmark_utils.benchmark_launcher \
    --mode=remote --backend=tpu_dist --benchmark=module --name=column_wise
```
