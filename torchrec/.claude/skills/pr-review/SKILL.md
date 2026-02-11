---
name: pr-review
description: Review TorchRec pull requests and diffs for code quality, test coverage, security, and backward compatibility. Use when reviewing PRs, diffs, or when asked to review code changes.
---

# TorchRec PR/Diff Review Skill

Review TorchRec pull requests and diffs focusing on what CI cannot check: code quality, test coverage adequacy, security vulnerabilities, and backward compatibility. Linting, formatting, type checking, and import ordering are handled by CI.

## Usage Modes

### No Argument

If the user invokes `/pr-review` with no arguments, use `get_local_changes` to review uncommitted changes or the current commit.

### Diff ID Mode

The user provides a Phabricator diff ID:

```
/pr-review D12345678
```

Use the `get_phabricator_diff_details` tool to fetch diff information.

### Local Changes Mode

Review uncommitted changes or the current commit:

```
/pr-review local
/pr-review branch
```

Use `get_local_changes` to get the diff data.

## Review Workflow

### Step 1: Fetch Information

1. For Phabricator diffs: Use `get_phabricator_diff_details` with `include_raw_diff=true`
2. For local changes: Use `get_local_changes`
3. Read related files for context using `read_file`

### Step 2: Analyze Changes

Read through the diff systematically:
1. Identify the purpose of the change from title/description
2. Group changes by type (new code, tests, config, docs)
3. Note the scope of changes (files affected, lines changed)

### Step 3: Deep Review

Perform thorough line-by-line analysis focusing on:

#### Code Quality
- Proper abstractions and design patterns
- Appropriate complexity levels
- Clear naming and documentation
- No code duplication

#### TorchRec-Specific Patterns
- Proper use of `KeyedJaggedTensor` and sparse tensors
- Correct sharding strategies (table-wise, row-wise, column-wise)
- Appropriate use of `DistributedModelParallel`
- Correct embedding table configurations
- Proper use of `EmbeddingBagCollection` and `EmbeddingCollection`

#### Testing
- Tests exist for new functionality
- Edge cases covered (empty tensors, single-device, multi-device)
- Tests follow TorchRec patterns (multi-process tests for distributed)
- Proper mocking of distributed primitives when needed

#### Performance
- No unnecessary tensor copies
- Efficient use of collectives
- Proper device placement
- Memory-efficient implementations

#### Backward Compatibility
- No breaking changes to public APIs
- Proper deprecation warnings if changing behavior
- Version compatibility considerations

### Step 4: Check BC Implications

For TorchRec, pay special attention to:
- Changes to `ShardingPlan` format
- Changes to `EmbeddingBagConfig` or `EmbeddingConfig`
- Changes to `KeyedJaggedTensor` structure
- Changes to distributed communication patterns
- Changes to serialization/deserialization

## Review Areas

| Area | Focus |
|------|-------|
| Code Quality | Abstractions, patterns, complexity |
| TorchRec Patterns | Sharding, embeddings, sparse tensors |
| Testing | Coverage, distributed tests, edge cases |
| Performance | Memory, communication, device handling |
| BC | Breaking changes, deprecation |

## Output Format

Structure your review as follows:

```markdown
## Review: D<number> / Local Changes

### Summary
Brief overall assessment of the changes (1-2 sentences).

### Code Quality
[Issues and suggestions, or "No concerns" if none]

### TorchRec Patterns
[Check for proper use of TorchRec APIs and patterns]

### Testing
- [ ] Tests exist for new functionality
- [ ] Edge cases covered
- [ ] Distributed scenarios tested (if applicable)
[Additional testing feedback]

### Performance
[Performance concerns if any, or "No performance concerns"]

### Backward Compatibility
[BC concerns if any, or "No BC-breaking changes"]

### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**

[Brief justification for recommendation]
```

### Specific Comments (Detailed Review Only)

When requested, add file-specific feedback with line references:

```markdown
### Specific Comments
- `torchrec/distributed/sharding.py:42` - Consider using table-wise sharding for small tables
- `torchrec/modules/embedding.py:100-105` - Missing test for empty input case
```

## Key Principles

1. **Focus on what CI cannot check** - Don't comment on formatting, linting, or type errors
2. **Be specific** - Reference file paths and line numbers
3. **Be actionable** - Provide concrete suggestions, not vague concerns
4. **Be proportionate** - Minor issues shouldn't block, but note them
5. **TorchRec expertise** - Apply knowledge of distributed embeddings and recommendation systems

## TorchRec Code Patterns to Check

### Embedding Configuration
```python
# Good: Explicit configuration
EmbeddingBagConfig(
    name="product",
    embedding_dim=64,
    num_embeddings=1000,
    feature_names=["product_id"],
    pooling=PoolingType.SUM,
)

# Check: Is pooling type appropriate?
# Check: Are feature names correct?
# Check: Is embedding_dim reasonable for the use case?
```

### Sharding Plans
```python
# Check: Is the sharding strategy appropriate for table size?
# Small tables -> TABLE_WISE
# Large tables -> ROW_WISE or COLUMN_WISE
# Check: Are compute kernels specified correctly?
```

### Distributed Tests
```python
# Check: Are multi-process tests using correct world_size?
# Check: Is cleanup handled properly?
# Check: Are results verified across all ranks?
```

## Files to Reference

When reviewing TorchRec code, consult:
- `torchrec/CLAUDE.md` - Coding style and testing patterns
- `torchrec/distributed/test_utils/` - Test utilities and patterns
- `torchrec/modules/` - Core module implementations
- `torchrec/distributed/planner/` - Sharding planner reference
