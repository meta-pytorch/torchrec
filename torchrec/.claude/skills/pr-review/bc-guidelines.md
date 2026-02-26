# TorchRec Backward Compatibility Guidelines

TorchRec is an open-source PyTorch domain library with external users. Any change to user-visible behavior is potentially BC-breaking and must be evaluated carefully.

## What Is a Public API in TorchRec

An API is **public** if:
- It lives outside the `fb/` directory
- Its name does not start with `_`
- It is exported via `__init__.py`
- It appears in public documentation or examples (`github/examples/`)

**Key public surfaces:**
- `torchrec.modules` - `EmbeddingBagCollection`, `EmbeddingCollection`, configs
- `torchrec.sparse` - `KeyedJaggedTensor`, `KeyedTensor`, `JaggedTensor`
- `torchrec.distributed` - `DistributedModelParallel`, sharders, `TrainPipelineSparseDist`, planner
- `torchrec.distributed.types` - `ShardingType`, `LazyAwaitable`, `ParameterSharding`
- `torchrec.optim` - `KeyedOptimizer`, fused optimizers
- `torchrec.metrics` - All `RecMetric` subclasses

**Note:** Code in `torchrec/fb/` is Meta-internal and not subject to OSS BC requirements, but still affects internal users.

## What Constitutes a BC-Breaking Change

### API Changes

| Change Type | BC Impact | Action Required |
|-------------|-----------|-----------------|
| Removing a public class/function | Breaking | Deprecation period required |
| Renaming a public API | Breaking | Deprecation period required |
| Removing/reordering function arguments | Breaking | Deprecation period required |
| Adding required arguments without defaults | Breaking | Add default value instead |
| Changing argument defaults | Potentially breaking | Document in release notes |
| Changing return type | Breaking | Deprecation period required |
| Changing `ShardingType` enum values | Breaking | Never change existing values; only add new ones |
| Modifying `EmbeddingBagConfig` fields | Potentially breaking | New fields must have defaults |

### Behavioral Changes

| Change Type | BC Impact | Action Required |
|-------------|-----------|-----------------|
| Changing KJT construction semantics | Breaking | KJT is the primary input format; changes affect all users |
| Changing default sharding strategy | Potentially breaking | May change performance characteristics silently |
| Changing embedding output layout | Breaking | Downstream models depend on output tensor shapes |
| New exceptions from existing APIs | Potentially breaking | Document and ensure it's expected |
| Changing planner heuristics | Potentially breaking | May produce different sharding plans for same inputs |

## Common BC Pitfalls in TorchRec

### 1. KeyedJaggedTensor Changes

KJT is the most widely used data structure. Changes are high-risk.

**Bad:**
```python
# Changing the meaning of lengths
# Before: lengths per feature per batch
# After: cumulative offsets
kjt = KeyedJaggedTensor(keys=keys, values=values, lengths=offsets)  # BREAKING
```

**Good:**
```python
# Add new parameter with default that preserves old behavior
kjt = KeyedJaggedTensor(
    keys=keys, values=values, lengths=lengths,
    offsets=offsets,  # New optional parameter
)
```

### 2. EmbeddingBagConfig Changes

**Bad:**
```python
# Adding a required field
@dataclass
class EmbeddingBagConfig:
    name: str
    embedding_dim: int
    num_embeddings: int
    new_required_field: str  # BREAKING - no default
```

**Good:**
```python
@dataclass
class EmbeddingBagConfig:
    name: str
    embedding_dim: int
    num_embeddings: int
    new_field: str = ""  # Has default, backward compatible
```

### 3. Sharding Type Additions

**Safe:** Adding a new `ShardingType` enum value is safe IF all switch/match statements have a default case or raise `NotImplementedError`.

**Unsafe:** If code uses exhaustive matching (explicit check for each variant), adding a new variant can cause runtime errors in user code.

### 4. DistributedModelParallel Signature

**Bad:**
```python
# Reordering args breaks positional callers
class DistributedModelParallel:
    def __init__(self, module, sharders, device):  # Was: module, device, sharders
```

**Good:**
```python
# New args at end with defaults
class DistributedModelParallel:
    def __init__(self, module, device, sharders=None, new_param=None):
```

### 5. Changing Module State Dict Keys

**Bad:**
```python
# Renaming internal parameters changes state_dict keys
# Before: model.embedding.weight
# After: model._embedding.weight  # BREAKING - saved models can't load
```

Changing state dict keys breaks model loading for all saved checkpoints. This requires explicit migration support.

## Deprecation Pattern

```python
import warnings

def old_function(x, old_arg=None, new_arg=None):
    if old_arg is not None:
        warnings.warn(
            "old_arg is deprecated and will be removed in a future release. "
            "Use new_arg instead.",
            FutureWarning,
            stacklevel=2,
        )
        new_arg = old_arg
    # ... rest of implementation
```

Use `FutureWarning` (not `DeprecationWarning`) for user-facing APIs so warnings are visible by default.

## When BC Breaks Are Acceptable

### With Deprecation
1. Deprecation warning added for at least one release
2. Migration path documented
3. Release notes updated

### Without Deprecation (Rare)
- Security vulnerabilities
- Serious bugs making the API unusable
- APIs explicitly marked experimental

## Review Checklist for BC

When reviewing a TorchRec diff, check:

- [ ] **No removed public APIs** - Or proper deprecation path exists
- [ ] **No changed signatures** - Or new args have defaults and are appended at the end
- [ ] **No changed defaults** - Or deprecation warning added
- [ ] **No changed return types/shapes** - Especially embedding output tensors
- [ ] **No changed KJT semantics** - Construction, splitting, permutation behavior preserved
- [ ] **No changed state dict keys** - Module parameter names unchanged
- [ ] **No changed ShardingType behavior** - Existing sharding types produce same results
- [ ] **No changed planner output** - Same inputs produce same sharding plans (or change is intentional and documented)
- [ ] **Deprecation uses FutureWarning** - Not `DeprecationWarning`
- [ ] **Deprecation has stacklevel=2** - Points to user code, not library internals
