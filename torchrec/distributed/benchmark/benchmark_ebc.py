#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Benchmark for module-level embedding lookup, one sharding type at a time.

``benchmark_runner`` is the per-rank entry point. It is invoked once per rank by the process
runner (``process_runner.run_single_process_func`` /
``process_runner.run_local_multi_process_func``), which owns process group init +
handshake and injects a live ``SingleProcessContext`` (``ctx``) plus this rank's
``rank`` and ``world_size``. The runner must therefore use ``ctx.device`` /
``ctx.pg`` directly rather than creating its own context.

Where ``benchmark_primitive`` times the bare collectives in isolation, this file times a
whole sharded ``EmbeddingBagCollection`` training iteration: the forward covers
``input_dist`` (the KJT A2A), the local TBE lookup, and ``output_dist`` (the pooled
embedding collective) together, and by default the backward and optimizer step follow
(``--backward=false`` measures the forward alone). The ``sharding_type`` flag picks how the tables are
sharded (see ``_SHARDING_GENERATORS``); with the table and input configuration otherwise
held fixed, the resulting latencies are directly comparable across sharding types. Only
latency is measured -- outputs are not checked for correctness.

Three sharding types are supported, and ``sharding_type`` may name one, several
(comma-separated) or ``"all"`` to run every one of them in sequence within the same job:

- ``table_wise``: each table lives whole on one rank (assigned round-robin), so
  ``output_dist`` is a ``PooledEmbeddingsAllToAll``.
- ``row_wise``: every table is row-sharded across all ranks, so each rank holds a slice
  of every table and ``output_dist`` becomes a ``PooledEmbeddingsReduceScatter``.
- ``column_wise``: every table is split column-wise across all ranks, so each rank holds
  a narrow slice of every table and ``output_dist`` is again an all-to-all, but of
  narrower per-rank embeddings.

A follow-up launcher binary will call ``runner`` explicitly with options to run on
MAST or locally.
"""

import gc
import itertools
import logging
import socket
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import torch

# Must precede the sharding_plan import below. sharding_plan imports
# planner.constants, which runs planner/__init__ -> planners -> enumerators, and
# enumerators imports back out of sharding_plan. Whichever of the two is imported
# first has to be the planner: entering through sharding_plan leaves it half-built
# when enumerators reaches back into it, which is an ImportError at startup.
import torchrec.distributed.planner  # noqa: F401
from torch import nn
from torchrec.distributed.benchmark.base import benchmark_func, BenchmarkResult
from torchrec.distributed.benchmark.utils import as_bool
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    ParameterShardingGenerator,
    row_wise,
    table_wise,
)
from torchrec.distributed.test_utils.process_runner import SingleProcessContext
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import (
    CombinedOptimizer,
    KeyedOptimizer,
    KeyedOptimizerWrapper,
)
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

if TYPE_CHECKING:
    from torchrec.distributed.train_pipeline import TrainPipelineSparseDist

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Learning rate for the dense-parameter optimizer. The benchmark never checks
# convergence, so the value only has to be finite -- it is the step's memory
# traffic that is being measured, not where it lands.
_LEARNING_RATE: float = 0.01

# Batches ``TrainPipelineSparseDist`` keeps in flight: one in forward/backward, one
# in input_dist, one being copied to device. A batch stream no longer than this
# never leaves fill/drain, so it is the floor for ``num_batches``.
_PIPELINE_DEPTH: int = 3

# Distinct batches allocated for a pipelined run, cycled to reach ``num_batches``.
# Bounds host memory independently of the iteration count. Must stay above
# ``_PIPELINE_DEPTH`` so a batch is never in flight in two stages at once -- that is
# the invariant to preserve if this is ever retuned.
_BATCH_POOL_SIZE: int = _PIPELINE_DEPTH + 1


# Sharding types this benchmark can run, each mapped to the factory that builds a
# table's ``ParameterShardingGenerator``. The factory is called per table with that
# table's index, the world size and the compute kernel. Only table-wise needs the
# index -- it pins a whole table to a single rank, so the index is what spreads the
# tables round-robin; row-wise and column-wise split every table across all ranks.
_SHARDING_GENERATORS: Dict[
    str, Callable[[int, int, str], ParameterShardingGenerator]
] = {
    ShardingType.TABLE_WISE.value: (
        lambda table_index, world_size, compute_kernel: table_wise(
            rank=table_index % world_size, compute_kernel=compute_kernel
        )
    ),
    ShardingType.ROW_WISE.value: (
        lambda table_index, world_size, compute_kernel: row_wise(
            compute_kernel=compute_kernel
        )
    ),
    ShardingType.COLUMN_WISE.value: (
        lambda table_index, world_size, compute_kernel: column_wise(
            ranks=list(range(world_size)), compute_kernel=compute_kernel
        )
    ),
}

# Special ``sharding_type`` token that expands to every entry in
# ``_SHARDING_GENERATORS`` (in registry order). Intentionally NOT a key in that dict
# -- it is expanded by :func:`parse_sharding_types`.
RUN_ALL: str = "all"


def available_sharding_types() -> List[str]:
    """Return the sorted sharding types accepted by the ``sharding_type`` flag.

    These (plus the special ``"all"`` token, :data:`RUN_ALL`) are the values
    :func:`parse_sharding_types` accepts.
    """
    return sorted(_SHARDING_GENERATORS)


def parse_sharding_types(value: Union[str, Sequence[str]]) -> List[str]:
    """Resolve the ``sharding_type`` selector into a concrete list of sharding types.

    Accepts a comma-separated string (e.g. ``"table_wise,row_wise"``) or a sequence of
    names. The special token ``"all"`` (:data:`RUN_ALL`) expands to every registered
    sharding type in registry order. Duplicates are dropped while preserving first-seen
    order.

    Returns:
        The ordered, de-duplicated list of sharding types to run (never empty).

    Raises:
        ValueError: if a token is neither a registered sharding type nor ``"all"``, or
            if the selector resolves to nothing.
    """
    if isinstance(value, str):
        tokens = [t.strip() for t in value.split(",") if t.strip()]
    else:
        tokens = [str(t).strip() for t in value if str(t).strip()]

    resolved: List[str] = []
    for tok in tokens:
        if tok == RUN_ALL:
            resolved.extend(_SHARDING_GENERATORS)
        elif tok in _SHARDING_GENERATORS:
            resolved.append(tok)
        else:
            raise ValueError(
                f"unknown sharding type {tok!r}; available: "
                f"{available_sharding_types()} (or {RUN_ALL!r} to run all of them)"
            )
    if not resolved:
        raise ValueError("sharding_type must select at least one sharding type")

    seen: set[str] = set()
    deduped: List[str] = []
    for sharding_type in resolved:
        if sharding_type not in seen:
            seen.add(sharding_type)
            deduped.append(sharding_type)
    return deduped


def _make_tables(
    num_tables: int,
    num_embeddings: int,
    embedding_dim: int,
) -> List[EmbeddingBagConfig]:
    """Build ``num_tables`` identically shaped tables, one feature each.

    Keeping every table the same shape makes the per-rank load balanced under any
    sharding type, so a latency difference between two runs is attributable to the
    sharding strategy rather than to a skewed table assignment.
    """
    return [
        EmbeddingBagConfig(
            name=f"table_{i}",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[f"feature_{i}"],
        )
        for i in range(num_tables)
    ]


def _make_input_kjt(
    tables: List[EmbeddingBagConfig],
    batch_size: int,
    pooling_factor: int,
    values_dtype: torch.dtype,
    device: torch.device,
) -> KeyedJaggedTensor:
    """Build this rank's local input batch, covering every feature in ``tables``.

    Every rank feeds the sharded module the full global key set for its own batch
    slice; ``input_dist`` is what routes each key to the rank that owns its table.
    Lengths are uniformly ``pooling_factor`` so no rank gets a heavier lookup than
    another. Indices are drawn per feature from its own table's row range -- unlike
    the pure-transport primitive benchmarks, the values here reach a real TBE
    kernel, so an out-of-range index would be a genuine error.

    ``KeyedJaggedTensor`` is key-major, so the values tensor is the per-key blocks
    of ``batch_size * pooling_factor`` indices concatenated in ``keys`` order.
    """
    keys: List[str] = []
    per_key_values: List[torch.Tensor] = []
    for table in tables:
        for feature_name in table.feature_names:
            keys.append(feature_name)
            per_key_values.append(
                torch.randint(
                    0,
                    table.num_embeddings,
                    (batch_size * pooling_factor,),
                    dtype=values_dtype,
                    device=device,
                )
            )

    lengths = torch.full(
        (len(keys) * batch_size,), pooling_factor, dtype=torch.int32, device=device
    )
    return KeyedJaggedTensor(
        keys=keys, values=torch.cat(per_key_values), lengths=lengths
    )


def _make_input_batch_pool(
    tables: List[EmbeddingBagConfig],
    batch_size: int,
    pooling_factor: int,
    values_dtype: torch.dtype,
    num_unique_batches: int,
) -> List[KeyedJaggedTensor]:
    """Build the bounded pool of host-resident batches the pipeline cycles over.

    The batches stay on CPU: the pipeline's first stage is the H2D copy, and it can
    only overlap that with compute if there is a copy left to do. Handing it batches
    already on the GPU would silently retire one of the three stages being measured.

    Only ``num_unique_batches`` of them are allocated, and
    :func:`_run_pipelined_iteration` cycles the pool to reach ``num_batches``. Host
    memory therefore scales with the pool, not with the iteration count -- a full
    batch here is ``num_tables * batch_size * pooling_factor`` indices, so
    materializing one per iteration is what exhausts host memory at large
    ``num_batches``.

    The pool size is fixed at :data:`_BATCH_POOL_SIZE` rather than exposed: it has to
    stay above the pipeline depth so a given batch is never in flight in two stages at
    once, and there is no reason to tune it per run. Note that cycling a small pool
    narrows the index footprint, which would flatter a caching compute kernel by
    keeping the working set warm in the TBE cache; raise the constant if that ever
    becomes the configuration under test.
    """
    return [
        _make_input_kjt(
            tables=tables,
            batch_size=batch_size,
            pooling_factor=pooling_factor,
            values_dtype=values_dtype,
            device=torch.device("cpu"),
        )
        for _ in range(num_unique_batches)
    ]


# @lint-ignore TORCHRECDOCSTRING Private benchmark adapter, not a public API to document.
class _PipelinedModel(nn.Module):
    """Adapts the sharded ``EmbeddingBagCollection`` to what the pipeline expects.

    ``TrainPipelineSparseDist`` calls ``losses, output = model(batch)`` and runs
    ``torch.sum(losses, dim=0).backward()`` itself, so the bare sharded module --
    which returns a ``LazyAwaitable[KeyedTensor]`` -- cannot be pipelined as-is.
    The loss is the per-row sum rather than a scalar because that backward reduces
    over dim 0, which a 0-dim tensor has not got.

    Wrapping also places the sharded module one level down, where
    ``_rewrite_model`` looks for it. ``forward`` hands the batch straight to it with
    no intervening op, which is the condition for the module to count as top-level
    and have its ``input_dist`` hoisted into the pipeline's own stage.
    """

    def __init__(self, ebc: ShardedEmbeddingBagCollection) -> None:
        super().__init__()
        self._ebc = ebc

    def forward(self, batch: KeyedJaggedTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        values = self._ebc(batch).values()
        return values.sum(dim=1), values


def _shard_ebc(
    tables: List[EmbeddingBagConfig],
    sharding_type: str,
    compute_kernel: str,
    world_size: int,
    device: torch.device,
    pg: torch.distributed.ProcessGroup,
) -> ShardedEmbeddingBagCollection:
    """Build an ``EmbeddingBagCollection`` and shard it with ``sharding_type``.

    The unsharded module is created on the ``meta`` device so no rank ever
    materializes the full tables; ``shard`` allocates only this rank's shards on
    ``device``.

    ``compute_kernel`` is passed explicitly rather than left to the plan builder's
    inference, which falls back to ``dense`` unless the table carries an in-backward
    optimizer -- that would benchmark ``nn.EmbeddingBag`` instead of the FBGEMM TBE
    a real sharded lookup runs on.
    """
    generator_factory = _SHARDING_GENERATORS[sharding_type]
    ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
    sharder = EmbeddingBagCollectionSharder()
    plan = construct_module_sharding_plan(
        ebc,
        {
            table.name: generator_factory(i, world_size, compute_kernel)
            for i, table in enumerate(tables)
        },
        # ModuleSharder is invariant in its module type, so the EBC sharder is not
        # assignable to the plan builder's ModuleSharder[nn.Module] parameter even
        # though it only calls module-agnostic methods on it.
        sharder=cast(ModuleSharder[nn.Module], sharder),
        world_size=world_size,
        device_type=device.type,
    )
    return sharder.shard(ebc, plan, ShardingEnv.from_process_group(pg), device)


def _make_optimizer(module: ShardedEmbeddingBagCollection) -> KeyedOptimizer:
    """Build the training-loop optimizer for a sharded ``EmbeddingBagCollection``.

    Mirrors the standard TorchRec construction: the module's own
    ``fused_optimizer`` for the in-backward (TBE-fused) parameters, combined with
    a plain SGD over whatever parameters are left. Under the default ``fused``
    compute kernel every embedding parameter is in-backward, so the dense set is
    empty and is skipped -- ``torch.optim.SGD`` rejects an empty parameter list.
    The ``dense`` kernel is the case that actually produces dense parameters.
    """
    optims: List[Union[KeyedOptimizer, Tuple[str, KeyedOptimizer]]] = [
        module.fused_optimizer
    ]
    dense_params: Dict[str, torch.Tensor] = dict(
        in_backward_optimizer_filter(module.named_parameters())
    )
    if dense_params:
        optims.append(
            KeyedOptimizerWrapper(
                dense_params,
                lambda params: torch.optim.SGD(params, lr=_LEARNING_RATE, foreach=True),
            )
        )
    return CombinedOptimizer(optims)


def _run_iteration(
    _batch_inputs: List[Any],
    *,
    module: nn.Module,
    kjt: KeyedJaggedTensor,
    optimizer: Optional[KeyedOptimizer],
) -> None:
    """One measured iteration: sharded-EBC forward, optionally backward + step.

    Rank alignment against the straggler effect is handled by ``PerfWrapper`` (it
    barriers before each iteration, outside the timing window); this function only
    runs the work. The forward returns a ``LazyAwaitable[KeyedTensor]``, so the
    ``.values()`` attribute access is what triggers ``output_dist``'s ``wait()``;
    it then only returns the dense tensor handle -- no data read, and on CUDA no
    host sync either (the collectives run async on the stream).

    ``optimizer is None`` selects forward-only. Otherwise the iteration is a full
    training step: ``values.sum()`` stands in for the dense arch, and its backward
    runs the ``output_dist`` collective in reverse plus the embedding backward.
    Under the ``fused`` compute kernel the TBE applies the weight update during that
    backward, so ``step()`` / ``zero_grad()`` never touch the embedding weights. They
    are not free, though, and calling them is not a formality:
    ``EmbeddingFusedOptimizer`` implements both as a push of the current learning rate
    into the TBE (a device-side ``learning_rate_tensor.fill_``), which is the channel
    LR scheduling reaches the kernel through. They also drive the dense parameters the
    ``dense`` compute kernel produces.

    We ``torch.cuda.synchronize()`` at the end to actually block the host on the
    whole chain inside the measured region; that is what makes the wall-clock timer
    reflect end-to-end latency (GPU-event timing is unaffected either way). The
    input KJT is reused across iterations.
    """
    if optimizer is not None:
        optimizer.zero_grad()
    values = module(kjt).values()
    if optimizer is not None:
        values.sum().backward()
        optimizer.step()
    if values.is_cuda:
        torch.cuda.synchronize(values.device)


def _run_pipelined_iteration(
    _batch_inputs: List[Any],
    *,
    pipeline: "TrainPipelineSparseDist[KeyedJaggedTensor, torch.Tensor]",
    batch_pool: List[KeyedJaggedTensor],
    num_batches: int,
    device: torch.device,
) -> None:
    """One measured iteration: drive ``TrainPipelineSparseDist`` over ``num_batches``.

    The stream is ``batch_pool`` cycled and cut to ``num_batches``, so the iteration
    count is independent of how many batches were actually allocated -- the pool is
    re-read rather than extended, which is what keeps host memory flat as
    ``num_batches`` grows.

    The measured unit is the whole pass, not a single ``progress()``. That is
    deliberate: ``PerfWrapper`` barriers between measured iterations, and a barrier
    between individual ``progress()`` calls would serialize exactly the copy /
    input_dist / compute overlap the pipeline exists to create. Dividing the reported
    latency by ``num_batches`` gives the per-batch cost; the reported QPS already
    accounts for it via ``sample_count``.

    ``progress()`` owns ``zero_grad()``, the backward and ``step()`` -- running them
    only while the model is training -- so unlike :func:`_run_iteration` this function
    must not call them, and forward-only is selected by handing it a model in eval
    mode rather than by branching here. It raises
    ``StopIteration`` once the iterator is drained and the in-flight batches have
    been executed (``execute_all_batches`` defaults to True), which is the loop's
    exit condition. ``reset()`` clears the batch/context queues so each measured
    iteration starts from an empty pipeline rather than inheriting the last one's
    tail.
    """
    pipeline.reset()
    dataloader = itertools.islice(itertools.cycle(batch_pool), num_batches)
    while True:
        try:
            pipeline.progress(dataloader)
        except StopIteration:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_sharding_type(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    sharding_type: str,
    **kwargs: Any,
) -> BenchmarkResult:
    """Run the module benchmark for a single sharding type.

    Shards an ``EmbeddingBagCollection`` with ``sharding_type`` over ``ctx.pg``, then
    measures the latency of one training iteration: the forward pass (``input_dist`` +
    local TBE lookup + ``output_dist``) and, unless ``backward`` is disabled, the
    backward and optimizer step as well. Correctness of the pooled output is
    intentionally not verified.

    Called once per selected sharding type by :func:`benchmark_runner`. The sharded
    module, its optimizer and the input batch are all local to this call, so they
    become collectable as soon as it returns -- which is what keeps a multi-sharding
    run from holding every module alive at once.

    Args:
        ctx: live single-process context (device + process group) injected by the
            process runner; use ``ctx.device`` / ``ctx.pg`` directly.
        rank: this process' global rank.
        world_size: total number of ranks.
        sharding_type: the (already validated) sharding type to run.
        **kwargs: benchmark options:
            compute_kernel (str): ``EmbeddingComputeKernel`` value the local lookup
                runs on. Default ``"fused"`` (the FBGEMM TBE); ``"dense"`` falls back
                to ``nn.EmbeddingBag``.
            backward (bool): measure a full training iteration -- ``values.sum()``
                as the stand-in dense arch, its backward, and the optimizer step --
                rather than the forward alone. Default True.
            pipeline (bool): drive the iteration through ``TrainPipelineSparseDist``,
                which overlaps the H2D copy and ``input_dist`` with compute, instead
                of running the module directly. Honours ``backward`` -- the pipeline
                skips its update when the model is in eval mode. Default False.
            num_batches (int): batches driven through the pipeline per measured
                iteration; must exceed the pipeline depth (3). Ignored unless
                ``pipeline``. Default 10.
            num_tables (int): number of embedding tables, one feature each. For
                table-wise they are assigned to ranks round-robin, so a multiple of
                ``world_size`` keeps the assignment balanced. Default 256.
            num_embeddings (int): rows per table. Default 1_000_000.
            embedding_dim (int): embedding width per table. Default 256.
            batch_size (int): this rank's local batch size. Default 4096.
            pooling_factor (int): indices looked up per feature per sample -- with
                the defaults each iteration looks up
                ``4096 * 256 * 20 ~= 21M`` embedding rows per rank. Default 20.
            values_dtype (torch.dtype): dtype of the KJT ``values`` tensor.
                Default int64.
            num_benchmarks (int): number of measured iterations. Default 100.
            num_profiles (int): number of profiled iterations (requires profile_dir).
                Default 5.
            profile_dir (str): directory for chrome traces; empty disables profiling.
            memory_snapshot (bool): capture a CUDA memory snapshot alongside the
                profile (requires profile_dir). Default True.
        The benchmark name (which keys the result files) is always
        ``"ebc_<sharding_type>"``: the launcher's ``name`` is the primitive-benchmark
        selector, so it is forwarded here as a list and deliberately not consumed.

    Returns:
        This rank's ``BenchmarkResult``.

    Raises:
        ValueError: if ``column_wise`` is selected but ``embedding_dim`` is not
            divisible by ``world_size``, or if ``pipeline`` is combined with a
            ``num_batches`` at or below the pipeline depth.
    """
    compute_kernel: str = str(
        kwargs.get("compute_kernel", EmbeddingComputeKernel.FUSED.value)
    )
    num_tables: int = int(kwargs.get("num_tables", 256))
    num_embeddings: int = int(kwargs.get("num_embeddings", 1_000_000))
    embedding_dim: int = int(kwargs.get("embedding_dim", 256))
    batch_size: int = int(kwargs.get("batch_size", 4096))
    pooling_factor: int = int(kwargs.get("pooling_factor", 20))
    values_dtype: torch.dtype = kwargs.get("values_dtype", torch.int64)
    backward: bool = as_bool(kwargs.get("backward"), True)
    pipeline: bool = as_bool(kwargs.get("pipeline"), False)
    num_batches: int = int(kwargs.get("num_batches", 10))
    num_benchmarks: int = int(kwargs.get("num_benchmarks", 100))
    num_profiles: int = int(kwargs.get("num_profiles", 5))
    profile_dir: str = str(kwargs.get("profile_dir", ""))
    memory_snapshot: bool = as_bool(kwargs.get("memory_snapshot"), True)

    name = f"ebc_{sharding_type}_pipelined" if pipeline else f"ebc_{sharding_type}"

    pg: Optional[torch.distributed.ProcessGroup] = ctx.pg
    assert pg is not None, "ctx.pg must be initialized by the process runner"

    if pipeline:
        # The pipeline holds three batches in flight, so a stream at or below that
        # depth is all fill and drain and never reaches the steady state the
        # benchmark is trying to measure.
        if num_batches <= _PIPELINE_DEPTH:
            raise ValueError(
                f"pipeline=True needs num_batches ({num_batches}) greater than the "
                f"pipeline depth ({_PIPELINE_DEPTH}) to reach steady state; "
                "use a larger --num_batches."
            )

    if sharding_type == ShardingType.COLUMN_WISE.value:
        # column_wise() splits the width evenly over the ranks it is given and
        # rejects a width it cannot divide, so fail here with the actionable message
        # rather than deep inside the plan builder.
        if embedding_dim % world_size != 0:
            raise ValueError(
                f"column_wise needs embedding_dim ({embedding_dim}) divisible by "
                f"world_size ({world_size}) to split each table evenly across ranks."
            )
    elif (
        sharding_type == ShardingType.TABLE_WISE.value and num_tables % world_size != 0
    ):
        logger.warning(
            "num_tables (%d) is not a multiple of world_size (%d): the round-robin "
            "table assignment is skewed, so the slowest rank bounds the measurement.",
            num_tables,
            world_size,
        )

    tables = _make_tables(
        num_tables=num_tables,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
    )
    module = _shard_ebc(
        tables=tables,
        sharding_type=sharding_type,
        compute_kernel=compute_kernel,
        world_size=world_size,
        device=ctx.device,
        pg=pg,
    )
    logger.info(
        "rank=%d local_rank=%d host=%s running module benchmark: sharding_type=%s "
        "compute_kernel=%s num_tables=%d num_embeddings=%d embedding_dim=%d "
        "batch_size=%d pooling_factor=%d backward=%s pipeline=%s num_batches=%d "
        "device=%s",
        rank,
        ctx.local_rank,
        socket.gethostname(),
        sharding_type,
        compute_kernel,
        num_tables,
        num_embeddings,
        embedding_dim,
        batch_size,
        pooling_factor,
        backward,
        pipeline,
        num_batches if pipeline else 1,
        ctx.device,
    )

    func_to_benchmark: Callable[..., None]
    if pipeline:
        batch_pool = _make_input_batch_pool(
            tables=tables,
            batch_size=batch_size,
            pooling_factor=pooling_factor,
            values_dtype=values_dtype,
            num_unique_batches=_BATCH_POOL_SIZE,
        )
        # Imported here, not at module scope: torchrec.distributed.train_pipeline
        # pulls in the planner, which imports sharding_plan -- already half-loaded by
        # this module's own import of it. Hoisting this to the top turns that latent
        # cycle into an ImportError on every run, pipelined or not.
        from torchrec.distributed.train_pipeline import TrainPipelineSparseDist

        pipelined_model = _PipelinedModel(module)
        if not backward:
            # progress() gates zero_grad / backward / step on ``model.training``, so
            # eval mode is how the pipeline itself expresses forward-only. The copy
            # and input_dist stages still overlap; only the update is dropped.
            pipelined_model.eval()

        func_to_benchmark = _run_pipelined_iteration
        benchmark_func_kwargs: Dict[str, Any] = {
            "pipeline": TrainPipelineSparseDist(
                model=pipelined_model,
                optimizer=_make_optimizer(module),
                device=ctx.device,
            ),
            "batch_pool": batch_pool,
            "num_batches": num_batches,
            "device": ctx.device,
        }
        # One measured iteration now drives the whole batch stream, so the QPS
        # denominator has to count every batch in it, not just one.
        sample_count = batch_size * num_batches
    else:
        func_to_benchmark = _run_iteration
        benchmark_func_kwargs = {
            "module": module,
            "kjt": _make_input_kjt(
                tables=tables,
                batch_size=batch_size,
                pooling_factor=pooling_factor,
                values_dtype=values_dtype,
                device=ctx.device,
            ),
            "optimizer": _make_optimizer(module) if backward else None,
        }
        sample_count = batch_size

    result = benchmark_func(
        name=name,
        rank=rank,
        world_size=world_size,
        func_to_benchmark=func_to_benchmark,
        bench_inputs=[],
        prof_inputs=[],
        benchmark_func_kwargs=benchmark_func_kwargs,
        num_profiles=num_profiles,
        num_benchmarks=num_benchmarks,
        profile_dir=profile_dir,
        memory_snapshot=memory_snapshot,
        device_type=ctx.device.type,
        # QPS is per-rank samples/s; multiply by world_size for the job-wide figure.
        sample_count=sample_count,
        pg=pg,
    )

    if rank == 0:
        logger.info("module benchmark result (%s):\n%s", name, result)

    return result


def benchmark_runner(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    **kwargs: Any,
) -> List[BenchmarkResult]:
    """Per-rank module benchmark entry point.

    Runs the sharded-``EmbeddingBagCollection`` benchmark once per selected sharding
    type and returns this rank's per-sharding-type results. ``sharding_type`` is
    resolved by :func:`parse_sharding_types`, so it may be a single name, a
    comma-separated list (e.g. ``"table_wise,row_wise"``), or ``"all"`` to run every
    registered sharding type. The selected runs happen sequentially in this one
    process, reusing the injected ``ctx`` (device + process group), ``rank`` and
    ``world_size``; the remaining ``kwargs`` are forwarded to each (see
    :func:`_benchmark_sharding_type`). Each run is dispatched under its own name
    (``ebc_<sharding_type>``), so their result files do not collide.

    Args:
        ctx: live single-process context (device + process group) injected by the
            process runner; use ``ctx.device`` / ``ctx.pg`` directly.
        rank: this process' global rank.
        world_size: total number of ranks.
        **kwargs: ``sharding_type`` (str | list) selects the run(s) (default
            ``"table_wise"``); the rest are forwarded to each selected run.

    Returns:
        This rank's per-sharding-type ``BenchmarkResult`` list, in resolved
        ``sharding_type`` order.
    """
    sharding_types = parse_sharding_types(
        kwargs.pop("sharding_type", ShardingType.TABLE_WISE.value)
    )

    logger.info(
        "rank=%d local_rank=%d host=%s running module benchmarks: %s",
        rank,
        ctx.local_rank,
        socket.gethostname(),
        sharding_types,
    )

    results: List[BenchmarkResult] = []
    for sharding_type in sharding_types:
        results.append(
            _benchmark_sharding_type(ctx, rank, world_size, sharding_type, **kwargs)
        )
        # Reclaim this run's shards before the next sharding type builds its own.
        # Both steps are needed: the sharded module sits in reference cycles (module
        # <-> autograd/state hooks), so dropping the last name does not free it until
        # a collection runs, and ``empty_cache`` only returns blocks that are already
        # unreferenced. Skipping either leaves the previous runs' tables resident and
        # inflates every later sharding type's peak memory -- which silently makes
        # the memory column of a multi-sharding run cumulative rather than per-run.
        gc.collect()
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()

    return results
