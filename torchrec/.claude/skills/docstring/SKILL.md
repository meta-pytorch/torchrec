---
name: docstring
description: Write docstrings for TorchRec functions and methods following PyTorch conventions. Use when writing or updating docstrings in TorchRec code.
---

# TorchRec Docstring Writing Guide

This skill describes how to write docstrings for functions and methods in the TorchRec project, following PyTorch conventions.

## General Principles

- Use **raw strings** (`r"""..."""`) for all docstrings to avoid issues with LaTeX/math backslashes
- Follow **Sphinx/reStructuredText** (reST) format for documentation
- Be **concise but complete** - include all essential information
- Always include **examples** when possible
- Use **cross-references** to related functions/classes

## Docstring Structure

### 1. Function Signature (First Line)

Start with the function signature showing all parameters:

```python
r"""function_name(param1, param2, *, kwarg1=default1, kwarg2=default2) -> ReturnType
```

**Notes:**
- Include the function name
- Show positional and keyword-only arguments (use `*` separator)
- Include default values
- Show return type annotation
- This line should NOT end with a period

### 2. Brief Description

Provide a one-line description of what the function does:

```python
r"""apply_optimizer_in_backward(optimizer_class, params, optimizer_kwargs) -> None

Applies optimizer to parameters in backward pass for memory efficiency.
```

### 3. Mathematical Formulas (if applicable)

Use Sphinx math directives for mathematical expressions:

```python
.. math::
    \text{output} = \text{input} \cdot \text{weight}^T
```

Or inline math: `:math:\`x^2\``

### 4. Cross-References

Link to related classes and functions using Sphinx roles:

- `:class:\`~torchrec.modules.EmbeddingBagCollection\`` - Link to a class
- `:func:\`torchrec.distributed.sharding.shard\`` - Link to a function
- `:meth:\`~Module.forward\`` - Link to a method
- `:attr:\`attribute_name\`` - Reference an attribute
- The `~` prefix shows only the last component

**Example:**
```python
See :class:`~torchrec.distributed.DistributedModelParallel` for details.
```

### 5. Notes and Warnings

Use admonitions for important information:

```python
.. note::
    This function requires CUDA to be available.

.. warning::
    This API is experimental and may change without notice.
```

### 6. Args Section

Document all parameters with type annotations and descriptions:

```python
Args:
    module (nn.Module): Module to be sharded across devices.
    device (torch.device, optional): Device to place the module. Default: ``None``
    sharders (List[ModuleSharder], optional): List of sharders to use for sharding.
        Default: ``None``
    plan (ShardingPlan, optional): Explicit sharding plan. If not provided, will be
        generated automatically. Default: ``None``
```

**Formatting rules:**
- Parameter name in **lowercase**
- Type in parentheses: `(Type)`, `(Type, optional)` for optional parameters
- Description follows the type
- For optional parameters, include "Default: ``value``" at the end
- Use double backticks for inline code: ``` ``None`` ```
- Indent continuation lines by 4 spaces

### 7. Returns Section

Document the return value:

```python
Returns:
    ShardedModule: The sharded module ready for distributed training.
        The module will have its parameters distributed according to
        the sharding plan.
```

### 8. Raises Section (if applicable)

Document exceptions that may be raised:

```python
Raises:
    ValueError: If the sharding plan is invalid for the given module.
    RuntimeError: If CUDA is not available when GPU sharding is requested.
```

### 9. Examples Section

Always include examples when possible:

```python
Examples::

    >>> import torchrec
    >>> from torchrec.modules import EmbeddingBagCollection
    >>> ebc = EmbeddingBagCollection(
    ...     tables=[
    ...         EmbeddingBagConfig(
    ...             name="product",
    ...             embedding_dim=64,
    ...             num_embeddings=1000,
    ...             feature_names=["product_id"],
    ...         ),
    ...     ],
    ... )
    >>> # Shard the module
    >>> sharded_ebc = shard(ebc, plan=plan)
```

**Formatting rules:**
- Use `Examples::` with double colon
- Use `>>>` prompt for Python code
- Include comments with `#` when helpful
- Show actual output when it helps understanding

## TorchRec-Specific Patterns

### Embedding Configuration

```python
Args:
    tables (List[EmbeddingBagConfig]): List of embedding table configurations.
        Each config specifies the table name, embedding dimension, number of
        embeddings, and feature names.
    device (Optional[torch.device]): Device to place embeddings. Default: ``None``
```

### Sharding-Related

```python
Args:
    sharding_type (ShardingType): How to shard the embedding table. Options are:
        - ``TABLE_WISE``: Each table on a single device
        - ``ROW_WISE``: Rows distributed across devices
        - ``COLUMN_WISE``: Columns distributed across devices
        - ``TABLE_ROW_WISE``: Combination of table and row sharding
```

### KeyedJaggedTensor

```python
Args:
    kjt (KeyedJaggedTensor): Sparse features in KeyedJaggedTensor format.
        Contains keys (feature names), values (embedding indices), and
        lengths/offsets for variable-length sequences.
```

## Complete Example

```python
def shard_modules(
    module: nn.Module,
    plan: ShardingPlan,
    env: ShardingEnv,
    device: Optional[torch.device] = None,
) -> nn.Module:
    r"""
    Shard a module's embedding tables according to a sharding plan.

    This function takes a module containing embedding tables and distributes
    them across multiple devices according to the provided sharding plan.
    It supports various sharding strategies including table-wise, row-wise,
    and column-wise sharding.

    Args:
        module (nn.Module): The module containing embedding tables to shard.
        plan (ShardingPlan): The sharding plan specifying how each table
            should be distributed.
        env (ShardingEnv): The sharding environment containing process group
            information and device topology.
        device (torch.device, optional): Target device for local shards.
            Default: ``None`` (uses current device)

    Returns:
        nn.Module: The sharded module with distributed embedding tables.

    Raises:
        ValueError: If the plan references tables not present in the module.
        RuntimeError: If the sharding environment is not properly initialized.

    .. note::
        This function modifies the module in-place for efficiency.

    .. warning::
        This is an experimental API and may change in future releases.

    Examples::

        >>> from torchrec.distributed import shard_modules
        >>> from torchrec.distributed.planner import EmbeddingShardingPlanner
        >>>
        >>> # Create a sharding plan
        >>> planner = EmbeddingShardingPlanner()
        >>> plan = planner.plan(module, sharders)
        >>>
        >>> # Shard the module
        >>> sharded_module = shard_modules(module, plan, env)
    """
    # implementation
```

## Quick Checklist

When writing a TorchRec docstring, ensure:

- [ ] Use raw string (`r"""`)
- [ ] Include function signature on first line
- [ ] Provide brief description
- [ ] Document all parameters in Args section with types
- [ ] Include default values for optional parameters
- [ ] Use Sphinx cross-references (`:func:`, `:class:`, `:meth:`)
- [ ] Add mathematical formulas if applicable
- [ ] Include at least one example in Examples section
- [ ] Add warnings/notes for experimental APIs
- [ ] Document any exceptions in Raises section
- [ ] Use proper math notation for tensor shapes
