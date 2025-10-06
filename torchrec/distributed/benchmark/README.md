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
