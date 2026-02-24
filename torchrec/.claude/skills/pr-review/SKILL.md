---
name: pr-review
description: Review TorchRec pull requests and diffs for distributed correctness, sharding safety, backward compatibility, and test coverage. Use when reviewing PRs, diffs, or when asked to review code changes.
allowed-tools: mcp__plugin_meta_mux__get_phabricator_diff_details, Bash(sl:*), Read, Grep, Glob, Task
---

# TorchRec PR/Diff Review Skill

Review TorchRec code changes focusing on what CI and linters cannot check: distributed correctness, sharding safety, backward compatibility, OSS boundary violations, and test adequacy. Linting, formatting, type checking, and import ordering are handled by CI.

## Usage Modes

### No Argument

If the user invokes `/pr-review` with no arguments, use `sl status` and `sl diff` (or `sl show .`) to review uncommitted changes or the current commit.

### Diff ID Mode

The user provides a Phabricator diff ID:

```
/pr-review D12345678
/pr-review D12345678 detailed
```

Use `mcp__plugin_meta_mux__get_phabricator_diff_details` with `include_raw_diff: true` and `include_diff_summary: true` to fetch the diff.

### Local Changes Mode

Review uncommitted changes or the current commit:

```
/pr-review local
/pr-review local detailed
```

1. Run `sl status` to check for uncommitted changes
2. Run `sl diff` for uncommitted changes, or `sl show .` for the current commit
3. For large diffs, use `sl diff --stat` first, then `Read` to examine specific files

## Review Workflow

### Step 1: Fetch and Understand Changes

1. Get the diff content (via Phabricator MCP or sl commands)
2. Read the diff summary/commit message to understand intent
3. Identify the scope: which TorchRec subsystems are affected (modules, distributed, sparse, metrics, fb/, etc.)

### Step 2: Classify Changes

Group changes by type:
- **Core logic** - New features, bug fixes, algorithm changes
- **Distributed/sharding** - Anything touching distributed/, sharding strategies, collectives
- **Public API** - Changes to interfaces, configs, module signatures
- **Tests** - New or modified tests
- **Internal** - Changes within fb/ (Meta-only)
- **Config/Build** - BUCK files, configs, dependencies

### Step 3: Deep Review

Perform thorough analysis using the review checklist. See [review-checklist.md](review-checklist.md) for detailed criteria covering:
- Distributed correctness and sharding safety
- FBGEMM integration
- Code quality and TorchRec patterns
- Testing adequacy
- Performance implications
- OSS boundary enforcement

### Step 4: Check Backward Compatibility

Evaluate BC implications for any change touching public APIs. See [bc-guidelines.md](bc-guidelines.md) for:
- What constitutes a BC-breaking change in TorchRec
- Public API boundaries (everything outside `fb/`)
- Required deprecation patterns
- Common BC pitfalls with KJT, EBC, DMP, and sharding types

### Step 5: Read Surrounding Code

For non-trivial changes, use `Read` to examine:
- The full file(s) being modified (not just the diff hunks)
- Related tests to verify coverage
- Similar implementations elsewhere in TorchRec for consistency

### Step 6: Formulate Review

Structure your review with actionable feedback organized by category.

## Review Areas

| Area | Focus | Reference |
|------|-------|-----------|
| Distributed Safety | Collectives, sharding, rank consistency, deadlocks | [review-checklist.md](review-checklist.md) |
| FBGEMM Integration | Kernel usage, op loading, config correctness | [review-checklist.md](review-checklist.md) |
| Code Quality | Patterns, abstractions, type hints | [review-checklist.md](review-checklist.md) |
| Testing | Coverage, distributed test patterns, edge cases | [review-checklist.md](review-checklist.md) |
| Performance | Memory, GPU utilization, pipeline efficiency | [review-checklist.md](review-checklist.md) |
| Backward Compatibility | Public API changes, deprecation | [bc-guidelines.md](bc-guidelines.md) |
| OSS Boundary | Public/internal separation | [review-checklist.md](review-checklist.md) |

## Output Format

Structure your review as follows:

```markdown
## Review: D<number> / Local Changes

### Summary
Brief overall assessment of the changes (1-2 sentences).

### Distributed Safety
[Issues with collectives, sharding, rank consistency. Or "No concerns" if none]

### Code Quality
[Issues and suggestions, or "No concerns" if none]

### Testing
- [ ] Tests exist for new functionality
- [ ] Distributed tests use MultiProcessTestBase
- [ ] Edge cases covered (empty KJT, single rank, etc.)
[Additional testing feedback]

### Backward Compatibility
[BC concerns if any, or "No BC-breaking changes"]

### Performance
[Performance concerns if any, or "No performance concerns"]

### OSS Boundary
[Boundary violations if any, or "No boundary issues"]

### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**

[Brief justification for recommendation]
```

### Specific Comments (Detailed Review Only)

**Only include this section if the user requests a "detailed" review.**

**Do not repeat observations already made in other sections.** This section is for additional file-specific feedback.

When requested, add file-specific feedback with line references:

```markdown
### Specific Comments
- `torchrec/distributed/sharding/rw_sharding.py:142` - This allreduce should use async_op=True to overlap with compute
- `torchrec/sparse/jagged_tensor.py:305-310` - Missing validation for empty lengths tensor
- `torchrec/distributed/tests/test_model_parallel.py:88` - Test only covers world_size=2, add world_size=4 case
```

## Key Principles

1. **No repetition** - Each observation appears in exactly one section
2. **Focus on what CI cannot check** - Don't comment on formatting, linting, or type errors
3. **Be specific** - Reference file paths and line numbers
4. **Be actionable** - Provide concrete suggestions, not vague concerns
5. **Be proportionate** - Minor issues shouldn't block, but note them
6. **Assume competence** - The author knows TorchRec and distributed systems; explain only non-obvious context
7. **Distributed correctness is paramount** - Subtle bugs in collectives cause hard-to-debug hangs and data corruption

## Files to Reference

When reviewing TorchRec code, consult:
- `torchrec/CLAUDE.md` - Coding style and testing patterns
- `torchrec/distributed/test_utils/` - Test utilities and patterns
- `torchrec/modules/` - Core module implementations
- `torchrec/distributed/planner/` - Sharding planner reference

## Instructions from User

<instructions>$ARGUMENTS</instructions>
