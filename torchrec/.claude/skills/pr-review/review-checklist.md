# TorchRec Review Checklist

This checklist covers areas that CI and linters cannot check. Skip items related to formatting, import ordering, and type checking (handled by `arc lint` and `arc pyre`).

## Distributed Correctness

### Collective Operations

- [ ] **Matching collectives** - Every `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all` is called by ALL ranks in the process group. Missing a collective on any rank causes a hang.
- [ ] **Consistent arguments** - Collective tensor shapes, dtypes, and op types match across ranks. Mismatches cause silent data corruption or NCCL errors.
- [ ] **Correct process group** - Operations use the intended process group, not the default. Check for `pg` vs `ctx.pg` vs `dist.group.WORLD` confusion.
- [ ] **Async ops completed** - Any `async_op=True` collective has a corresponding `.wait()` before the result is consumed.
- [ ] **No rank-conditional collectives** - Code like `if rank == 0: dist.all_reduce(...)` is almost always wrong. All ranks must participate.

### Sharding Safety

- [ ] **All ShardingType variants handled** - When switching on `ShardingType`, all 7 variants are covered: `TABLE_WISE`, `ROW_WISE`, `COLUMN_WISE`, `TABLE_ROW_WISE`, `TABLE_COLUMN_WISE`, `GRID_SHARD`, `DATA_PARALLEL`. Missing a variant causes silent fallthrough or runtime errors.
- [ ] **Shard metadata consistency** - `ShardedTensorMetadata` (sizes, offsets, placements) is consistent across ranks. Inconsistencies cause incorrect lookups.
- [ ] **Correct device placement** - Tensors created during sharding are on the expected device. Watch for accidental CPU tensors in GPU sharding paths.
- [ ] **LazyAwaitable usage** - `_wait_impl` returns the correct tensor. The awaitable is not `.wait()`'d prematurely (which defeats pipelining).
- [ ] **Input distribution correctness** - `KJT` splitting/redistribution preserves feature keys ordering and lengths-values correspondence.

### Pipeline Safety

- [ ] **Pipeline stage ordering** - `TrainPipelineSparseDist` stages (sparse data dist, forward, backward) maintain correct data flow dependencies.
- [ ] **No eager waits in pipeline** - Operations that should be deferred via `LazyAwaitable` are not eagerly awaited, which would serialize communication and computation.
- [ ] **Fused optimizer consistency** - Fused optimizer parameters match the sharded embedding tables. Adding/removing tables without updating the optimizer causes silent training bugs.

## FBGEMM Integration

- [ ] **Correct kernel selection** - `SplitTableBatchedEmbeddingBagsCodegen` for training, `IntNBitTableBatchedEmbeddingBagsCodegen` for quantized inference. Using the wrong one causes incorrect results.
- [ ] **Op loading guards** - FBGEMM ops loaded with try/except: `torch.ops.load_library(...)` wrapped in `try/except OSError`.
- [ ] **Config alignment** - `EmbeddingLocation`, `ComputeDevice`, `PoolingMode`, `SparseType` values are consistent between TorchRec configs and FBGEMM kernel configs.
- [ ] **Batch size handling** - FBGEMM kernels have specific requirements for batch size alignment. Verify batch dimensions are correct.

## Code Quality

### TorchRec Patterns

- [ ] **Match existing patterns** - Code follows architectural patterns already in TorchRec (ABC interfaces, dataclass configs, enum patterns)
- [ ] **Config via dataclass** - Configuration uses `@dataclass` with typed fields and defaults, not dicts or kwargs
- [ ] **Enum for variants** - Enumerated options use `@unique class MyEnum(Enum)`, not string constants
- [ ] **ABC for interfaces** - Module interfaces inherit from both `abc.ABC` and `nn.Module`
- [ ] **No dynamic attribute access** - Avoid `setattr`/`getattr` for state management; use explicit class members

### Code Clarity

- [ ] **Self-explanatory code** - Variable and function names convey intent
- [ ] **Useful comments only** - Comments explain non-obvious context (e.g., why a specific sharding order matters), not what the code does
- [ ] **No backward-compatibility hacks** - Unused code is deleted, not renamed with underscores
- [ ] **Appropriate complexity** - Solutions are as simple as possible for the current requirements

### Common Issues to Flag

- Magic numbers without explanation (especially embedding dimensions, batch sizes, world sizes)
- Copy-pasted sharding logic that should be a shared helper
- Overly defensive error handling for impossible cases
- `Optional` parameters that are never actually `None` in practice

## Testing

### Test Existence

- [ ] **Tests exist** - New functionality has corresponding tests
- [ ] **Tests in correct location** - Tests are at `$(dirname)/test(s)/` following TorchRec convention
- [ ] **Distributed code has distributed tests** - Any change to distributed/ requires `MultiProcessTestBase` tests

### Distributed Test Patterns

- [ ] **Uses MultiProcessTestBase** - Distributed tests inherit from `MultiProcessTestBase` and use `_run_multi_process_test`
- [ ] **Test function signature** - Test callable follows `_test_func(rank: int, world_size: int, **kwargs) -> None` pattern
- [ ] **MultiProcessContext** - Test function uses `with MultiProcessContext(rank, world_size, backend="gloo") as ctx:` for setup
- [ ] **Multiple world sizes** - Tests cover at least `world_size=2`; ideally also `world_size=4` for sharding strategies that behave differently
- [ ] **Backend selection** - Tests use `gloo` backend (not `nccl`) unless specifically testing GPU collective behavior

### Test Quality

- [ ] **Edge cases covered** - Empty KJT (no features), single-rank world, zero-length embeddings, single-feature tables
- [ ] **Error conditions tested** - Expected exceptions tested with `assertRaises` or `assertRaisesRegex`
- [ ] **Sharding type coverage** - If testing a sharder, relevant `ShardingType` variants are tested
- [ ] **Deterministic** - Tests set seeds via `seed_and_log` from `torchrec.test_utils`, no flaky assertions on floating point equality without tolerance

### Common Testing Issues

- Tests that only check `world_size=2` for sharding strategies that partition differently at higher world sizes
- Distributed tests missing cleanup (process group destruction)
- Tests that hardcode CUDA device without checking availability
- Missing tests for the interaction between pipeline stages

## Performance

### Memory

- [ ] **No unnecessary tensor copies** - Avoid `.clone()`, `.contiguous()`, or `.to(device)` in hot paths when not needed
- [ ] **Gradient memory** - Proper use of `torch.no_grad()` and `.detach()` to avoid retaining computation graphs
- [ ] **Embedding table memory** - New tables or increased dimensions accounted for in memory estimates
- [ ] **KJT materialization** - Avoid materializing dense tensors from sparse KJT representations unnecessarily

### GPU Utilization

- [ ] **Overlap communication and compute** - Pipelined operations use `async_op=True` where possible
- [ ] **No unnecessary synchronization** - Avoid `torch.cuda.synchronize()` or `.item()` in training loops
- [ ] **Batch size efficiency** - FBGEMM kernels perform best with specific batch size alignments

### Common Performance Issues

- Creating new tensors inside training loops instead of pre-allocating
- Synchronous collective operations where async would work
- Redundant `all_to_all` operations that could be batched
- KJT permutation/splitting done eagerly when it could be deferred

## OSS Boundary

- [ ] **No public-to-internal imports** - Code outside `fb/` must NOT import from `torchrec/fb/` or any `fb/` subdirectory
- [ ] **License headers** - Public files use BSD license header, not Meta confidential
- [ ] **`# pyre-strict`** - All Python files include `# pyre-strict` after the copyright header
- [ ] **Internal features gated** - If a public module has internal extensions, they are registered via the `fb/` module's `__init__.py`, not hard-coded in public code
- [ ] **No internal dependencies in public BUCK targets** - Public build targets should not depend on `//torchrec/fb/...` targets
