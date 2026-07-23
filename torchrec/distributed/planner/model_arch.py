#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Extract the replay-minimal model-architecture surface for reproduction.

The planner never consumes model weights -- only the tables it shards, which it
discovers the same way the enumerator does: for each module a sharder claims
(matched by ``sharder_name(sharder.module_type)``) the tables are
``sharder.shardable_parameters(module)`` -- NOT ``embedding_bag_configs()``,
which misses tables held by ads arch modules (``PooledEmbeddingArch`` etc. store
raw ``nn.EmbeddingBag`` and expose only ``tables()``). Each table also records
its owning sharder type, since the sharder determines the table's sharding /
storage and replay must place it under the same sharder. Capturing this surface
(``ModelArch``) is enough to rebuild a meta-device module and re-plan; its
content hash (``model_arch_hash``) is the model axis of content-addressing for
the reproduction storage layer.
"""

import hashlib
import json
from typing import cast, Dict, Iterable, List, Optional, Tuple

import torch.nn as nn
from torchrec.distributed.planner.types import ModelArch, SharderArch, TableArch
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.types import ModuleSharder


def _enum_value(value: object) -> Optional[str]:
    """Enum -> its ``.value`` (stable across processes); str() otherwise; None -> None."""
    if value is None:
        return None
    # 3-arg getattr (with default): pyre special-cases getattr(x, "literal") as
    # attribute access and would flag object.value [16]; the 3-arg form returns
    # the default type instead, so it type-checks without narrowing ``object``.
    return getattr(value, "value", None) if hasattr(value, "value") else str(value)


def _sharder_arch(sharder: ModuleSharder[nn.Module]) -> SharderArch:
    """Capture the plan-affecting optimizer subset from a sharder's fused_params.

    The optimizer (and its scalar hyperparams) drives the storage multiplier and
    has no ParameterConstraints equivalent, so it must be persisted for faithful
    replay. Caching/prefetch/dtype knobs are deliberately NOT captured here --
    they are already reproducible from the persisted constraints.
    """
    fp = dict(getattr(sharder, "fused_params", None) or {})
    return SharderArch(
        sharder_type=type(sharder).__name__,
        optimizer=_enum_value(fp.get("optimizer")),
        learning_rate=fp.get("learning_rate"),
        eps=fp.get("eps"),
        weight_decay=fp.get("weight_decay"),
        weight_decay_mode=_enum_value(fp.get("weight_decay_mode")),
        beta1=fp.get("beta1"),
        beta2=fp.get("beta2"),
    )


def _table_config(module: nn.Module, name: str) -> Optional[object]:
    """Best-effort per-table config for ``name`` (feature_names/pooling/data_type).

    EBC/EC expose ``embedding_bag_configs()`` / ``embedding_configs()``; ads arch
    modules (``PooledEmbeddingArch`` etc.) expose ``tables()`` instead. Try each
    and return the matching config, else None.
    """
    for accessor in ("embedding_bag_configs", "embedding_configs", "tables"):
        fn = getattr(module, accessor, None)
        if not callable(fn):
            continue
        try:
            # cast: after callable() narrowing the checker types fn() as ``object``
            # (not iterable); the accessor returns a config sequence at runtime.
            for cfg in cast(Iterable[object], fn()):
                if getattr(cfg, "name", None) == name:
                    return cfg
        except Exception:
            continue
    return None


def extract_model_arch(
    module: nn.Module,
    sharders: List[ModuleSharder[nn.Module]],
) -> ModelArch:
    """Collect the tables the planner shards, keyed by (name, owning sharder).

    Mirrors ``EmbeddingEnumerator.enumerate``: BFS the module tree, and for each
    module a sharder claims (``sharder_name(type(module))`` in the sharder map)
    take its tables from ``sharder.shardable_parameters(module)`` and do NOT
    descend into it. Table shape comes from the parameter; feature_names/pooling/
    data_type from the module's config accessor. Deduped by (name, sharder_type),
    sorted for a deterministic, content-addressable result. Best-effort throughout
    (this is observability, not the plan).
    """
    sharder_map: Dict[str, ModuleSharder[nn.Module]] = {
        sharder_name(sharder.module_type): sharder for sharder in sharders
    }
    tables: Dict[Tuple[str, str], TableArch] = {}
    queue: List[nn.Module] = [module]
    while queue:
        child = queue.pop()
        sharder = sharder_map.get(sharder_name(type(child)))
        if sharder is None:
            queue.extend(child.children())
            continue
        # Claimed module: take its tables, do not descend (mirrors the enumerator).
        sharder_type = type(sharder).__name__
        try:
            params = dict(sharder.shardable_parameters(child))
        except Exception:
            continue
        for name, param in params.items():
            key = (name, sharder_type)
            if not name or key in tables:
                continue
            cfg = _table_config(child, name)
            shape = getattr(param, "shape", None)
            num_embeddings = (
                int(shape[0])
                if shape is not None and len(shape) >= 1
                else int(getattr(cfg, "num_embeddings", 0) or 0)
            )
            embedding_dim = (
                int(shape[1])
                if shape is not None and len(shape) >= 2
                else int(getattr(cfg, "embedding_dim", 0) or 0)
            )
            tables[key] = TableArch(
                name=name,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                feature_names=(
                    tuple(getattr(cfg, "feature_names", ()) or ()) if cfg else ()
                ),
                pooling=_enum_value(getattr(cfg, "pooling", None)) if cfg else None,
                data_type=_enum_value(getattr(cfg, "data_type", None)) if cfg else None,
                sharder_type=sharder_type,
            )
    sorted_tables = tuple(tables[key] for key in sorted(tables))
    sharder_types = tuple(sorted({type(sharder).__name__ for sharder in sharders}))
    # One record per distinct sharder config (APS stamps the same fused_params on
    # every sharder, so this is typically 1-3 records), sorted for determinism.
    sharder_archs = tuple(
        sorted(
            {_sharder_arch(sharder) for sharder in sharders},
            key=lambda s: (s.sharder_type, s.optimizer or ""),
        )
    )
    return ModelArch(
        tables=sorted_tables,
        sharder_types=sharder_types,
        sharders=sharder_archs,
    )


def model_arch_hash(arch: ModelArch) -> str:
    """Stable 16-char content hash of a ModelArch (the model axis key).

    Serialized with sorted keys so identical architectures hash identically
    across processes, independent of dict/attr ordering.
    """
    payload = {
        "tables": [
            {
                "name": t.name,
                "num_embeddings": t.num_embeddings,
                "embedding_dim": t.embedding_dim,
                "feature_names": list(t.feature_names),
                "pooling": t.pooling,
                "data_type": t.data_type,
                "sharder_type": t.sharder_type,
            }
            for t in arch.tables
        ],
        "sharder_types": list(arch.sharder_types),
        "sharders": [
            {
                "sharder_type": s.sharder_type,
                "optimizer": s.optimizer,
                "learning_rate": s.learning_rate,
                "eps": s.eps,
                "weight_decay": s.weight_decay,
                "weight_decay_mode": s.weight_decay_mode,
                "beta1": s.beta1,
                "beta2": s.beta2,
            }
            for s in arch.sharders
        ],
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


# Bytes per element for the data_type strings captured on TableArch (the torchrec
# DataType ``.value``). Unknown/absent data_type falls back to fp32 (4 bytes).
_DTYPE_BYTES: Dict[str, float] = {
    "FP32": 4.0,
    "FP16": 2.0,
    "BF16": 2.0,
    "INT8": 1.0,
    "UINT8": 1.0,
    "INT4": 0.5,
    "INT2": 0.25,
}


def total_sparse_param_bytes(arch: ModelArch) -> int:
    """Total sparse (embedding) parameter size in bytes across all tables.

    The sparse counterpart to ``report_metadata.total_model_param_size`` (all
    params, in bytes): ``sum(num_embeddings * embedding_dim * element_size)`` over
    the captured tables, where element_size comes from each table's ``data_type``
    (fp32 fallback when unknown/None). Reported by the persistence sinks as the
    model's sparse footprint.
    """
    total = 0
    for t in arch.tables:
        element_bytes = _DTYPE_BYTES.get((t.data_type or "").upper(), 4.0)
        total += int(t.num_embeddings * t.embedding_dim * element_bytes)
    return total
