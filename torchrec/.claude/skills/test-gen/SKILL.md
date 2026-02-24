---
name: test-gen
argument-hint: [file-path or "local"]
description: Generate tests for TorchRec source files with correct patterns (unit, distributed, hypothesis), proper BUCK targets, and test utilities. Use when asked to generate tests, add test coverage, or write tests for a module.
allowed-tools: Read, Write, Edit, Bash(sl:*), Bash(buck2:*), Grep, Glob, Task
---

# TorchRec Test Generator

Generate idiomatic TorchRec tests by reading source files, detecting the appropriate test type, scaffolding test code with correct patterns, and creating/updating BUCK targets.

## Usage Modes

### File Path Mode

```
/test-gen torchrec/distributed/sharding/my_sharder.py
/test-gen torchrec/modules/new_module.py
```

Generate tests for the specified source file.

### Auto-Detect Mode

```
/test-gen local
/test-gen
```

Detect changed files via `sl status` and generate tests for new/modified source files that lack test coverage.

## Workflow

### Phase 1: Identify Source Files

**File path mode:** Read the specified file.

**Auto-detect mode:**
1. Run `sl status` to find changed files
2. Filter to `.py` source files in `torchrec/` (exclude test files, `__init__.py`, BUCK files)
3. For each source file, check if a corresponding test file exists at `$(dirname)/test(s)/test_$(basename)`
4. Present the list of untested files and ask the user which to generate tests for

### Phase 2: Analyze Source Code

Read the source file and classify it:

**Detection rules (in priority order):**

1. **Distributed test** if ANY of:
   - File is under `torchrec/distributed/`
   - Imports from `torch.distributed`, `torchrec.distributed`, or uses `ProcessGroup`
   - Defines sharders, sharded modules, or uses `ShardingType`
   - Uses `LazyAwaitable`, `all_to_all`, `all_reduce`, `all_gather`

2. **Hypothesis-parameterized test** if ANY of:
   - Source defines enums, configs, or strategies with multiple variants
   - Source handles multiple `ShardingType` or `EmbeddingComputeKernel` values
   - Source has branching behavior based on config parameters

3. **Unit test** (default) if:
   - File is under `torchrec/modules/`, `torchrec/sparse/`, `torchrec/optim/`, `torchrec/metrics/`
   - No distributed primitives used

A file can be **both** distributed AND hypothesis-parameterized.

Extract from the source file:
- Public classes and their methods
- Public functions
- Constructor signatures and required arguments
- Key data types (KJT, EBC, KeyedTensor, etc.)
- Dependencies and imports needed for tests

### Phase 3: Determine Test Location

Follow TorchRec convention:
- Source: `torchrec/foo/bar/my_module.py`
- Test: `torchrec/foo/bar/tests/test_my_module.py`

If a `tests/` directory doesn't exist, create it.
If a test file already exists, **add new test methods** rather than overwriting.

### Phase 4: Generate Test Code

Generate tests following the patterns below. See [test-patterns.md](test-patterns.md) for complete templates.

**For all test types:**
- BSD license header + `# pyre-strict`
- Type hints on all methods (return `-> None` for test methods)
- Use `self.assertEqual`, `self.assertTrue`, `torch.testing.assert_close` for assertions
- Cover: happy path, edge cases (empty inputs, single element), error conditions
- Name tests descriptively: `test_<what>_<condition>`

**For unit tests:**
- Inherit from `unittest.TestCase`
- Test each public method/function independently
- For modules: test `forward()` with representative inputs, verify output shapes and types

**For distributed tests:**
- Inherit from `MultiProcessTestBase`
- Use `@staticmethod` or module-level `_test_func(rank, world_size, **kwargs)` pattern
- Wrap per-rank logic in `with MultiProcessContext(rank, world_size, backend) as ctx:`
- Default `world_size=2`, add `world_size=4` for sharding tests
- Use `backend="gloo"` unless testing GPU-specific behavior
- Add `@unittest.skipIf(torch.cuda.device_count() < N, "Not enough GPUs...")` for CUDA tests

**For hypothesis tests:**
- Add `@given(...)` with `st.sampled_from([...])` for enum/config parameters
- Add `@settings(verbosity=Verbosity.verbose, max_examples=N, deadline=None)`
- Use `assume()` to filter invalid parameter combinations
- Keep `max_examples` reasonable (4-8 for distributed tests, 10-20 for unit tests)

### Phase 5: Create/Update BUCK Target

Read the existing BUCK file in the `tests/` directory (or create one if it doesn't exist).

**For CPU-only unit tests:**
```python
python_unittest(
    name = "test_my_module",
    srcs = ["test_my_module.py"],
    deps = [
        "//caffe2:_torch",
        # ... source deps ...
    ],
)
```

**For GPU/distributed tests:**
```python
python_unittest(
    name = "test_my_module",
    srcs = ["test_my_module.py"],
    remote_execution = re_test_utils.remote_execution(
        mig = "false",
        platform = "gpu-remote-execution",
        resource_units = 2,
    ),
    deps = [
        "//caffe2:_torch",
        "//torchrec/distributed/test_utils:multi_process",
        # ... source deps ...
    ],
)
```

**If hypothesis is used, add:**
```python
    supports_static_listing = False,
```
and add to deps:
```python
    "fbsource//third-party/pypi/hypothesis:hypothesis",
```

**BUCK rules:**
- Use `load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")` for standard tests
- Add `load("@fbcode_macros//build_defs/lib:re_test_utils.bzl", "re_test_utils")` for GPU tests
- Include `oncall("torchrec")` if already present in the BUCK file
- Derive deps from the test file's imports — map each `torchrec.*` import to its BUCK target by checking the source directory's BUCK file

### Phase 6: Verify

1. Ask the user to review the generated test file
2. Suggest running the test:
   ```
   buck2 test fbcode//torchrec/path/to/tests:test_my_module
   ```
3. If hypothesis is used, suggest running with more examples:
   ```
   buck2 test fbcode//torchrec/path/to/tests:test_my_module -- -s
   ```

## Test Utilities Reference

Use these utilities when generating tests:

| Utility | Import | When to Use |
|---------|--------|-------------|
| `MultiProcessTestBase` | `torchrec.distributed.test_utils.multi_process` | All distributed tests |
| `MultiProcessContext` | `torchrec.distributed.test_utils.multi_process` | Per-rank setup/teardown |
| `ModelInput` | `torchrec.distributed.test_utils.test_model` | Generating test inputs for models |
| `TestSparseNN` | `torchrec.distributed.test_utils.test_model` | Test model with embedding tables |
| `sharding_single_rank_test` | `torchrec.distributed.test_utils.test_sharding` | Testing sharders |
| `create_test_sharder` | `torchrec.distributed.test_utils.test_sharding` | Creating test sharder instances |
| `skip_if_asan_class` | `torchrec.test_utils` | Skip entire class under ASAN |
| `seed_and_log` | `torchrec.test_utils` | Deterministic seeding with logging |
| `get_free_port` | `torchrec.test_utils` | Getting available port for dist init |

## Constraints

- NEVER overwrite existing test methods. Add new methods to existing test classes or create new classes.
- NEVER add tests for private methods (starting with `_`) unless they contain complex logic that's critical to test.
- ALWAYS match the import style of the source file (modern `list[str]` vs `List[str]`).
- ALWAYS check if similar tests already exist before generating duplicates.
- Keep generated tests focused and minimal — don't test framework behavior or trivial getters/setters.

## Instructions from User

<instructions>$ARGUMENTS</instructions>
