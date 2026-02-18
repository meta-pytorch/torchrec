#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Purpose:
This module provides decorator functions to annotate hardware-specific properties and communication patterns on
HardwarePerfConfig classes, enabling flexible and hardware-aware performance modeling for distributed training in TorchRec.

Key Features:
Hardware Property Annotations: Use decorators like @hbm_mem_bw, @ddr_mem_bw, and @bwd_compute_multiplier to
specify hardware characteristics (such as memory bandwidth and compute multipliers) on config classes.
Sharding-Type Specific Overrides: Communication decorators (@fwd_comms, @bwd_comms, @input_dist_comms) can be
applied to methods, optionally filtered by sharding type (e.g., TABLE_WISE, ROW_WISE). This enables custom
performance formulas for specific sharding strategies, while using defaults for others.
Usage Example: You can annotate a config class with hardware properties and override communication formulas
for particular sharding types.
Flexible Method Resolution: When an estimator calls a communication method, it checks for sharding-type-specific
overrides. If a method is annotated for a specific sharding type and matches the context, it uses the custom method; otherwise, it falls back to the default implementation. This keeps code DRY and allows targeted optimizations.

Typical Use Cases:
Modeling hardware performance for distributed training.
Customizing communication cost formulas for different sharding strategies.
Creating hardware-specific estimators for Meta’s infrastructure.

"""

from typing import Any, Callable, cast, List, Optional, overload, Type, TypeVar, Union

# Type variable for decorator return types
T = TypeVar("T")

# =============================================================================
# Helper Functions for Accessing Annotated Methods
# =============================================================================


def _matches_sharding_type(method: Callable[..., float], sharding_type: str) -> bool:
    """
    Check if a custom method matches the given sharding type.

    Args:
        method: The custom method to check
        sharding_type: The sharding type to match against (e.g., 'table_wise')

    Returns:
        True if the method applies to this sharding type
    """
    custom_sharding_type = getattr(method, "_custom_sharding_type", None)
    if custom_sharding_type is None:
        # No sharding type restriction - applies to all
        return True

    # Handle list/tuple of sharding types
    if isinstance(custom_sharding_type, (list, tuple)):
        return sharding_type.lower() in [st.lower() for st in custom_sharding_type]

    # Single sharding type - normalize both to lowercase for comparison
    return custom_sharding_type.lower() == sharding_type.lower()


def _get_annotated_method(
    config: Any,
    annotation_attr: str,
    sharding_type: Optional[str] = None,
) -> Optional[Callable[..., float]]:
    """
    Scan all methods on config for the annotation attribute.

    This function finds any method that has the specified annotation attribute,
    regardless of the method name. The annotation is the source of truth.

    Args:
        config: The hardware config to check
        annotation_attr: The annotation attribute to look for (e.g., '_is_custom_output_write_size')
        sharding_type: Optional sharding type to filter by. If specified, only returns
            the method if it applies to this sharding type.

    Returns:
        The annotated method if found and matches sharding type, otherwise None.
    """
    for attr_name in dir(config):
        if attr_name.startswith("_"):
            continue
        attr = getattr(config, attr_name, None)
        if attr and callable(attr) and getattr(attr, annotation_attr, False):
            # Cast to the expected return type
            method = cast(Callable[..., float], attr)
            # Check sharding type if provided
            if sharding_type is not None:
                if not _matches_sharding_type(method, sharding_type):
                    continue
            return method
    return None


def get_custom_method(
    obj: Any,
    method_name: str,
    annotation_attr: str,
    sharding_type: Optional[str] = None,
) -> Optional[Callable[..., float]]:
    """
    Get a method from an object if it has the specified annotation attribute
    and optionally matches the specified sharding type.

    Args:
        obj: The object to get the method from
        method_name: Name of the method to retrieve
        annotation_attr: The annotation attribute to check for
        sharding_type: Optional sharding type to filter by. If provided, the method
            must either have no sharding_type restriction or match this sharding_type.

    Returns:
        The method if it exists, has the annotation, and matches the sharding type
        (if specified), otherwise None
    """

    method = getattr(obj, method_name, None)
    if method and getattr(method, annotation_attr, False):
        # Check sharding type if provided
        if sharding_type is not None:
            if not _matches_sharding_type(method, sharding_type):
                return None
        return method
    return None


def get_forward_compute(
    config: Any, sharding_type: Optional[str] = None
) -> Optional[Callable[..., float]]:
    """
    Get custom forward compute method if annotated with @forward_compute.

    Scans all methods on the config for the annotation, regardless of method name.

    Args:
        config: The hardware config to check
        sharding_type: Optional sharding type to filter by. If specified, only returns
            the method if it applies to this sharding type.
    """
    return _get_annotated_method(config, "_is_custom_forward_compute", sharding_type)


def get_backward_compute(
    config: Any, sharding_type: Optional[str] = None
) -> Optional[Callable[..., float]]:
    """
    Get custom backward compute method if annotated with @backward_compute.

    Scans all methods on the config for the annotation, regardless of method name.

    Args:
        config: The hardware config to check
        sharding_type: Optional sharding type to filter by. If specified, only returns
            the method if it applies to this sharding type.
    """
    return _get_annotated_method(config, "_is_custom_backward_compute", sharding_type)


def get_prefetch_compute(config: Any) -> Optional[Callable[..., float]]:
    """Get custom prefetch compute method if annotated with @prefetch_compute."""
    return _get_annotated_method(config, "_is_custom_prefetch_compute")


def get_input_dist_comms(
    config: Any, sharding_type: Optional[str] = None
) -> Optional[Callable[..., float]]:
    """
    Get custom input dist comms method if annotated with @input_dist_comms.

    Scans all methods on the config for the annotation, regardless of method name.

    Args:
        config: The hardware config to check
        sharding_type: Optional sharding type to filter by. If specified, only returns
            the method if it applies to this sharding type.
    """
    return _get_annotated_method(config, "_is_custom_input_dist_comms", sharding_type)


def get_fwd_comms(
    config: Any, sharding_type: Optional[str] = None
) -> Optional[Callable[..., float]]:
    """
    Get custom fwd comms method if annotated with @fwd_comms.

    Scans all methods on the config for the annotation, regardless of method name.

    Args:
        config: The hardware config to check
        sharding_type: Optional sharding type to filter by. If specified, only returns
            the method if it applies to this sharding type.
    """
    return _get_annotated_method(config, "_is_custom_fwd_comms", sharding_type)


def get_bwd_comms(
    config: Any, sharding_type: Optional[str] = None
) -> Optional[Callable[..., float]]:
    """
    Get custom bwd comms method if annotated with @bwd_comms.

    Scans all methods on the config for the annotation, regardless of method name.

    Args:
        config: The hardware config to check
        sharding_type: Optional sharding type to filter by. If specified, only returns
            the method if it applies to this sharding type.
    """
    return _get_annotated_method(config, "_is_custom_bwd_comms", sharding_type)


def get_output_write_size(
    config: Any, sharding_type: Optional[str] = None
) -> Optional[Callable[..., float]]:
    """
    Get custom output write size method if annotated with @output_write_size.

    Scans all methods on the config for the annotation, regardless of method name.
    The annotation is the source of truth, not the method name.

    Args:
        config: The hardware config to check
        sharding_type: Optional sharding type to filter by. If specified, only returns
            the method if it applies to this sharding type.
    """
    return _get_annotated_method(config, "_is_custom_output_write_size", sharding_type)


# =============================================================================
# SECTION 1: Bandwidth Decorators
# =============================================================================


def hbm_mem_bw(value: float) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to set HBM memory bandwidth for a hardware config.

    HBM High Bandwidth Memory

    Args:
        value: HBM memory bandwidth in bytes/second

    Example:
        @hbm_mem_bw(6200 * 1024 * 1024 * 1024)  # 6200 GB/s
        class MyHardwarePerfConfig(DefaultHardwarePerfConfig):
            pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls.hbm_mem_bw = value  # pyre-ignore[16]
        return cls

    return decorator


def ddr_mem_bw(value: float) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to set DDR memory bandwidth for a hardware config.

    DDR memory is the system/host memory.

    Args:
        value: DDR memory bandwidth in bytes/second

    Example:
        @ddr_mem_bw(100 * 1024 * 1024 * 1024)  # 100 GB/s
        class MyHardwarePerfConfig(DefaultHardwarePerfConfig):
            pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls.ddr_mem_bw = value  # pyre-ignore[16]
        return cls

    return decorator


def hbm_to_ddr_mem_bw(value: float) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to set HBM to DDR memory bandwidth for a hardware config.

    This is the bandwidth for UVM (Unified Virtual Memory) operations
    that move data between GPU HBM and system DDR memory.

    Args:
        value: HBM to DDR bandwidth in bytes/second

    Example:
        @hbm_to_ddr_mem_bw(25.6 * 1024 * 1024 * 1024)  # 25.6 GB/s
        class MyHardwarePerfConfig(DefaultHardwarePerfConfig):
            pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls.hbm_to_ddr_mem_bw = value  # pyre-ignore[16]
        return cls

    return decorator


def device_bw(
    bandwidth: Optional[float] = None,
    *,
    device: Optional[str] = None,
    compute_kernel: Optional[Union[str, List[str]]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to set device bandwidth for a hardware config.

    Usage:
        # General device bandwidth (existing behavior)
        @device_bw(3200 * 1024 * 1024 * 1024)
        class MyConfig(HardwarePerfConfig): pass

        # Specific device + kernel bandwidth
        @device_bw(bandwidth=5000, device='cuda', compute_kernel='fused')
        @device_bw(bandwidth=3000, device='cuda', compute_kernel='dense')
        class MyConfig(HardwarePerfConfig): pass

        # Multiple compute kernels with same bandwidth (NEW)
        @device_bw(bandwidth=5000, device='cuda', compute_kernel=['fused', 'fused_uvm'])
        class MyConfig(HardwarePerfConfig): pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        if device is not None and compute_kernel is not None:
            # Bind to local variables to satisfy type narrowing
            device_str: str = device
            compute_kernel_val: str | List[str] = compute_kernel
            # Specific device + kernel override
            # IMPORTANT: Check if kernel_device_bandwidths is inherited from parent class
            # If so, create a new dictionary for this class to avoid modifying the shared one
            if (
                not hasattr(cls, "kernel_device_bandwidths")
                or cls.kernel_device_bandwidths is None
            ):
                cls.kernel_device_bandwidths = {}  # pyre-ignore[16]
            else:
                # Check if we're using an inherited dictionary by comparing with parent's
                # If so, create a new copy for this class
                for base in cls.__mro__[1:]:
                    if (
                        hasattr(base, "kernel_device_bandwidths")
                        and cls.kernel_device_bandwidths
                        is base.kernel_device_bandwidths
                    ):
                        cls.kernel_device_bandwidths = dict(
                            cls.kernel_device_bandwidths
                        )  # pyre-ignore[16]
                        break

            # Normalize compute_kernel to a list for uniform processing
            kernels = (
                compute_kernel_val
                if isinstance(compute_kernel_val, list)
                else [compute_kernel_val]
            )

            # Store with lowercase keys for case-insensitive lookup
            for kernel in kernels:
                key = (device_str.lower(), kernel.lower())
                cls.kernel_device_bandwidths[key] = bandwidth  # pyre-ignore[16]
        else:
            # General device bandwidth (existing behavior)
            cls.device_bw = bandwidth  # pyre-ignore[16]
        return cls

    return decorator


# =============================================================================
# SECTION 4: Communication Bandwidth Decorators
# =============================================================================


def intra_host_bw(value: float) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to set intra-host communication bandwidth.

    This is the bandwidth for communication between GPUs within the
    same host (e.g., NVLink, NVSwitch).

    Args:
        value: Intra-host bandwidth in bytes/second

    Example:
        @intra_host_bw(600 * 1024 * 1024 * 1024)  # 600 GB/s (NVLink)
        class MyHardwarePerfConfig(DefaultHardwarePerfConfig):
            pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls.intra_host_bw = value  # pyre-ignore[16]
        return cls

    return decorator


def inter_host_bw(value: float) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to set inter-host communication bandwidth.

    This is the bandwidth for communication between GPUs across
    different hosts (e.g., InfiniBand, RoCE).

    Args:
        value: Inter-host bandwidth in bytes/second

    Example:
        @inter_host_bw(25 * 1024 * 1024 * 1024)  # 25 GB/s (IB HDR)
        class MyHardwarePerfConfig(DefaultHardwarePerfConfig):
            pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls.inter_host_bw = value  # pyre-ignore[16]
        return cls

    return decorator


# =============================================================================
# SECTION 3: Custom Compute Method Decorators
# =============================================================================


@overload
def forward_compute(
    method_or_sharding_type: Callable[..., float],
) -> Callable[..., float]: ...


@overload
def forward_compute(
    method_or_sharding_type: str,
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


@overload
def forward_compute(
    method_or_sharding_type: List[str],
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


@overload
def forward_compute(
    *,
    sharding_type: Union[str, List[str]],
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


def forward_compute(
    method_or_sharding_type: Optional[
        Union[Callable[..., float], str, List[str]]
    ] = None,
    *,
    sharding_type: Optional[Union[str, List[str]]] = None,
) -> Union[
    Callable[..., float], Callable[[Callable[..., float]], Callable[..., float]]
]:
    """
    Decorator to mark a method as custom forward compute implementation.

    Use this decorator when a hardware config needs custom logic for
    computing forward pass time instead of the default linear regression model.
    This is useful for hardware with unique kernel breakdown patterns
    (e.g., Athena's BCI + IRLE model).
    Example:
        class AthenaHardwarePerfConfig(DefaultHardwarePerfConfig):
            # Applies to all sharding types
            @forward_compute
            def compute_fwd(self, ctx: ShardPerfContext) -> float:
                # Custom Athena kernel breakdown logic
                bci_time = self._compute_bci(ctx)
                irle_time = self._compute_irle(ctx)
                return self.bci_coeff * bci_time + self.irle_coeff * irle_time

            # Applies only to TABLE_WISE sharding
            @forward_compute(sharding_type=ShardingType.TABLE_WISE.value)
            def compute_fwd(self, ctx: ShardPerfContext) -> float:
                return self._compute_table_wise_fwd(ctx)
    """

    def _apply_decorator(
        method: Callable[..., float],
        target_sharding_type: Optional[Union[str, List[str]]],
    ) -> Callable[..., float]:
        method._is_custom_forward_compute = True  # pyre-ignore[16]
        method._custom_sharding_type = target_sharding_type  # pyre-ignore[16]
        return method

    # Case 1: @forward_compute (no parentheses)
    if callable(method_or_sharding_type):
        return _apply_decorator(method_or_sharding_type, None)

    # Case 2: @forward_compute(sharding_type='table_wise')
    if sharding_type is not None:
        target_sharding_type = sharding_type

        def decorator(method: Callable[..., float]) -> Callable[..., float]:
            return _apply_decorator(method, target_sharding_type)

        return decorator

    # Case 3: @forward_compute('table_wise') or @forward_compute(['table_wise', 'column_wise']) - positional argument
    if isinstance(method_or_sharding_type, (str, list)):
        target_sharding_type = method_or_sharding_type

        def decorator(method: Callable[..., float]) -> Callable[..., float]:
            return _apply_decorator(method, target_sharding_type)

        return decorator

    # Case 4: @forward_compute() - empty parentheses
    def decorator(method: Callable[..., float]) -> Callable[..., float]:
        return _apply_decorator(method, None)

    return decorator


@overload
def backward_compute(
    method_or_sharding_type: Callable[..., float],
) -> Callable[..., float]: ...


@overload
def backward_compute(
    method_or_sharding_type: str,
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


@overload
def backward_compute(
    method_or_sharding_type: List[str],
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


@overload
def backward_compute(
    *,
    sharding_type: Union[str, List[str]],
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


def backward_compute(
    method_or_sharding_type: Optional[
        Union[Callable[..., float], str, List[str]]
    ] = None,
    *,
    sharding_type: Optional[Union[str, List[str]]] = None,
) -> Union[
    Callable[..., float], Callable[[Callable[..., float]], Callable[..., float]]
]:
    """
    Decorator to mark a method as custom backward compute implementation.
    """

    def _apply_decorator(
        method: Callable[..., float],
        target_sharding_type: Optional[Union[str, List[str]]],
    ) -> Callable[..., float]:
        method._is_custom_backward_compute = True  # pyre-ignore[16]
        method._custom_sharding_type = target_sharding_type  # pyre-ignore[16]
        return method

    # Case 1: @backward_compute (no parentheses)
    if callable(method_or_sharding_type):
        return _apply_decorator(method_or_sharding_type, None)

    # Case 2: @backward_compute(sharding_type='table_wise')
    if sharding_type is not None:
        target_sharding_type = sharding_type

        def decorator(method: Callable[..., float]) -> Callable[..., float]:
            return _apply_decorator(method, target_sharding_type)

        return decorator

    # Case 3: @backward_compute('table_wise') or @backward_compute(['table_wise', 'column_wise']) - positional argument
    if isinstance(method_or_sharding_type, (str, list)):
        target_sharding_type = method_or_sharding_type

        def decorator(method: Callable[..., float]) -> Callable[..., float]:
            return _apply_decorator(method, target_sharding_type)

        return decorator

    # Case 4: @backward_compute() - empty parentheses
    def decorator(method: Callable[..., float]) -> Callable[..., float]:
        return _apply_decorator(method, None)

    return decorator


def prefetch_compute(
    method: Callable[..., float],
) -> Callable[..., float]:
    """
    Decorator to mark a method as custom prefetch compute implementation.

    Use this decorator when a hardware config needs custom logic for
    computing prefetch/cache loading time.

    Example:
        class CustomHardwarePerfConfig(DefaultHardwarePerfConfig):
            @prefetch_compute
            def compute_prefetch(self, ctx: ShardPerfContext, expected_cache_fetches: float) -> float:
                # Custom prefetch logic
                return expected_cache_fetches * ctx.shard_embedding_dim / self.custom_prefetch_bw
    """
    method._is_custom_prefetch_compute = True  # pyre-ignore[16]
    return method


@overload
def input_dist_comms(
    method_or_sharding_type: Callable[..., float],
) -> Callable[..., float]: ...


@overload
def input_dist_comms(
    method_or_sharding_type: str,
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


@overload
def input_dist_comms(
    method_or_sharding_type: List[str],
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


@overload
def input_dist_comms(
    *,
    sharding_type: Union[str, List[str]],
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


def input_dist_comms(
    method_or_sharding_type: Optional[
        Union[Callable[..., float], str, List[str]]
    ] = None,
    *,
    sharding_type: Optional[Union[str, List[str]]] = None,
) -> Union[
    Callable[..., float], Callable[[Callable[..., float]], Callable[..., float]]
]:
    """
    Decorator to mark a method as custom input distribution communication implementation.

    Use this decorator when a hardware config needs custom logic for
    computing input distribution communication time (A2A input dist latency).

    Example:
        class CustomHardwarePerfConfig(DefaultHardwarePerfConfig):
            # Applies to all sharding types
            @input_dist_comms
            def compute_input_dist_comms(self, ctx: ShardPerfContext) -> float:
                return ctx.input_read_size / self.custom_input_dist_bw

            # Applies only to TABLE_WISE sharding
            @input_dist_comms(sharding_type=ShardingType.TABLE_WISE.value)
            def compute_input_dist_comms(self, ctx: ShardPerfContext) -> float:
                return ctx.input_read_size / self.table_wise_input_dist_bw
    """

    def _apply_decorator(
        method: Callable[..., float],
        target_sharding_type: Optional[Union[str, List[str]]],
    ) -> Callable[..., float]:
        method._is_custom_input_dist_comms = True  # pyre-ignore[16]
        method._custom_sharding_type = target_sharding_type  # pyre-ignore[16]
        return method

    # Case 1: @input_dist_comms (no parentheses)
    if callable(method_or_sharding_type):
        return _apply_decorator(method_or_sharding_type, None)

    # Case 2: @input_dist_comms(sharding_type='table_wise')
    if sharding_type is not None:
        target_sharding_type = sharding_type

        def decorator(method: Callable[..., float]) -> Callable[..., float]:
            return _apply_decorator(method, target_sharding_type)

        return decorator

    # Case 3: @input_dist_comms('table_wise') or @input_dist_comms(['table_wise', 'column_wise']) - positional argument
    if isinstance(method_or_sharding_type, (str, list)):
        target_sharding_type = method_or_sharding_type

        def decorator(method: Callable[..., float]) -> Callable[..., float]:
            return _apply_decorator(method, target_sharding_type)

        return decorator

    # Case 4: @input_dist_comms() - empty parentheses
    def decorator(method: Callable[..., float]) -> Callable[..., float]:
        return _apply_decorator(method, None)

    return decorator


@overload
def fwd_comms(
    method_or_sharding_type: Callable[..., float],
) -> Callable[..., float]: ...


@overload
def fwd_comms(
    method_or_sharding_type: str,
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


@overload
def fwd_comms(
    method_or_sharding_type: List[str],
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


@overload
def fwd_comms(
    *,
    sharding_type: Union[str, List[str]],
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


def fwd_comms(
    method_or_sharding_type: Optional[
        Union[Callable[..., float], str, List[str]]
    ] = None,
    *,
    sharding_type: Optional[Union[str, List[str]]] = None,
) -> Union[
    Callable[..., float], Callable[[Callable[..., float]], Callable[..., float]]
]:
    """
    Decorator to mark a method as custom forward communication implementation.

    Use this decorator when a hardware config needs custom logic for
    computing forward pass communication time (e.g., All-to-all, Reduce-scatter).

    Example:
        class CustomHardwarePerfConfig(DefaultHardwarePerfConfig):
            # Applies to all sharding types
            @fwd_comms
            def compute_fwd_comms(self, ctx: ShardPerfContext) -> float:
                return ctx.fwd_output_write_size / self.custom_fwd_comms_bw

            # Applies only to TABLE_WISE sharding
            @fwd_comms(sharding_type=ShardingType.TABLE_WISE.value)
            def compute_fwd_comms(self, ctx: ShardPerfContext) -> float:
                return ctx.fwd_output_write_size / self.table_wise_fwd_comms_bw
    """

    def _apply_decorator(
        method: Callable[..., float],
        target_sharding_type: Optional[Union[str, List[str]]],
    ) -> Callable[..., float]:
        method._is_custom_fwd_comms = True  # pyre-ignore[16]
        method._custom_sharding_type = target_sharding_type  # pyre-ignore[16]
        return method

    # Case 1: @fwd_comms (no parentheses)
    if callable(method_or_sharding_type):
        return _apply_decorator(method_or_sharding_type, None)

    # Case 2: @fwd_comms(sharding_type='table_wise')
    if sharding_type is not None:
        target_sharding_type = sharding_type

        def decorator(method: Callable[..., float]) -> Callable[..., float]:
            return _apply_decorator(method, target_sharding_type)

        return decorator

    # Case 3: @fwd_comms('table_wise') or @fwd_comms(['table_wise', 'column_wise']) - positional argument
    if isinstance(method_or_sharding_type, (str, list)):
        target_sharding_type = method_or_sharding_type

        def decorator(method: Callable[..., float]) -> Callable[..., float]:
            return _apply_decorator(method, target_sharding_type)

        return decorator

    # Case 4: @fwd_comms() - empty parentheses
    def decorator(method: Callable[..., float]) -> Callable[..., float]:
        return _apply_decorator(method, None)

    return decorator


@overload
def bwd_comms(
    method_or_sharding_type: Callable[..., float],
) -> Callable[..., float]: ...


@overload
def bwd_comms(
    method_or_sharding_type: str,
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


@overload
def bwd_comms(
    method_or_sharding_type: List[str],
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


@overload
def bwd_comms(
    *,
    sharding_type: Union[str, List[str]],
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


def bwd_comms(
    method_or_sharding_type: Optional[
        Union[Callable[..., float], str, List[str]]
    ] = None,
    *,
    sharding_type: Optional[Union[str, List[str]]] = None,
) -> Union[
    Callable[..., float], Callable[[Callable[..., float]], Callable[..., float]]
]:
    """
    Decorator to mark a method as custom backward communication implementation.

    Use this decorator when a hardware config needs custom logic for
    computing backward pass communication time (e.g., All-gather, All-reduce).

    Example:
        class CustomHardwarePerfConfig(DefaultHardwarePerfConfig):
            # Applies to all sharding types
            @bwd_comms
            def compute_bwd_comms(self, ctx: ShardPerfContext) -> float:
                return ctx.bwd_output_write_size / self.custom_bwd_comms_bw

            # Applies only to ROW_WISE sharding
            @bwd_comms(sharding_type=ShardingType.ROW_WISE.value)
            def compute_bwd_comms(self, ctx: ShardPerfContext) -> float:
                return ctx.bwd_output_write_size / self.row_wise_bwd_comms_bw
    """

    def _apply_decorator(
        method: Callable[..., float],
        target_sharding_type: Optional[Union[str, List[str]]],
    ) -> Callable[..., float]:
        method._is_custom_bwd_comms = True  # pyre-ignore[16]
        method._custom_sharding_type = target_sharding_type  # pyre-ignore[16]
        return method

    # Case 1: @bwd_comms (no parentheses)
    if callable(method_or_sharding_type):
        return _apply_decorator(method_or_sharding_type, None)

    # Case 2: @bwd_comms(sharding_type='row_wise')
    if sharding_type is not None:
        target_sharding_type = sharding_type

        def decorator(method: Callable[..., float]) -> Callable[..., float]:
            return _apply_decorator(method, target_sharding_type)

        return decorator

    # Case 3: @bwd_comms('row_wise') or @bwd_comms(['row_wise', 'column_wise']) - positional argument
    if isinstance(method_or_sharding_type, (str, list)):
        target_sharding_type = method_or_sharding_type

        def decorator(method: Callable[..., float]) -> Callable[..., float]:
            return _apply_decorator(method, target_sharding_type)

        return decorator

    # Case 4: @bwd_comms() - empty parentheses
    def decorator(method: Callable[..., float]) -> Callable[..., float]:
        return _apply_decorator(method, None)

    return decorator


@overload
def output_write_size(
    method_or_sharding_type: Callable[..., float],
) -> Callable[..., float]: ...


@overload
def output_write_size(
    method_or_sharding_type: str,
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


@overload
def output_write_size(
    method_or_sharding_type: List[str],
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


@overload
def output_write_size(
    *,
    sharding_type: Union[str, List[str]],
) -> Callable[[Callable[..., float]], Callable[..., float]]: ...


def output_write_size(
    method_or_sharding_type: Optional[
        Union[Callable[..., float], str, List[str]]
    ] = None,
    *,
    sharding_type: Optional[Union[str, List[str]]] = None,
) -> Union[
    Callable[..., float], Callable[[Callable[..., float]], Callable[..., float]]
]:
    """
    Decorator to mark a method as custom output write size implementation.

    Use this decorator when a hardware config needs to use a different data type
    size for fwd_output_write_size calculation. For example, FB legacy estimators
    use output_data_type_size instead of fwd_a2a_comm_data_type_size.

    The method should have signature:
        def get_output_write_size(self, ctx: ShardPerfContext, is_fwd: bool = True) -> float

    Example:
        class FBHardwarePerfConfig(HardwarePerfConfig):
            # Applies to all sharding types
            @output_write_size
            def get_output_write_size(self, ctx: ShardPerfContext, is_fwd: bool = True) -> float:
                # FB legacy uses output_data_type_size for compute formula
                return ctx.batch_outputs * ctx.world_size * ctx.emb_dim * ctx.output_data_type_size

            # Applies only to TABLE_WISE sharding
            @output_write_size(sharding_type=ShardingType.TABLE_WISE.value)
            def get_output_write_size(self, ctx: ShardPerfContext, is_fwd: bool = True) -> float:
                return ctx.batch_outputs * ctx.world_size * ctx.emb_dim * ctx.output_data_type_size
    """

    def _apply_decorator(
        method: Callable[..., float],
        target_sharding_type: Optional[Union[str, List[str]]],
    ) -> Callable[..., float]:
        method._is_custom_output_write_size = True  # pyre-ignore[16]
        method._custom_sharding_type = target_sharding_type  # pyre-ignore[16]
        return method

    # Case 1: @output_write_size (no parentheses)
    if callable(method_or_sharding_type):
        return _apply_decorator(method_or_sharding_type, None)

    # Case 2: @output_write_size(sharding_type='table_wise')
    if sharding_type is not None:
        target_sharding_type = sharding_type

        def decorator(method: Callable[..., float]) -> Callable[..., float]:
            return _apply_decorator(method, target_sharding_type)

        return decorator

    # Case 3: @output_write_size('table_wise') or @output_write_size(['table_wise', 'column_wise']) - positional argument
    if isinstance(method_or_sharding_type, (str, list)):
        target_sharding_type = method_or_sharding_type

        def decorator(method: Callable[..., float]) -> Callable[..., float]:
            return _apply_decorator(method, target_sharding_type)

        return decorator

    # Case 4: @output_write_size() - empty parentheses
    def decorator(method: Callable[..., float]) -> Callable[..., float]:
        return _apply_decorator(method, None)

    return decorator


# =============================================================================
# SECTION 4: Strategy Hook Decorators
# =============================================================================


def use_min_dim_for_lookup(value: bool = True) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to enable/disable min_dim constraint for embedding lookup size.

    When enabled (True), embedding lookup uses max(emb_dim, 32) for kernel
    efficiency. This is the TABLE_WISE behavior per OSS EmbeddingPerfEstimator.

    Args:
        value: Whether to use min_dim constraint (default: True)

    Example:
        @use_min_dim_for_lookup(True)
        class MyHardwarePerfConfig(DefaultHardwarePerfConfig):
            pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls._use_min_dim_for_lookup = value  # pyre-ignore[16]
        return cls

    return decorator


def use_block_usage_penalty(value: bool = True) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to enable/disable block usage penalty for forward compute.

    When enabled (True), forward compute is multiplied by a penalty factor
    based on emb_dim alignment with GPU block sizes. This is the TABLE_WISE
    behavior per OSS EmbeddingPerfEstimator.

    Penalty factors:
    - emb_dim >= 128: 1.0 (no penalty)
    - emb_dim >= 64: HALF_BLOCK_PENALTY
    - emb_dim >= 32: QUARTER_BLOCK_PENALTY

    Args:
        value: Whether to apply block usage penalty (default: True)

    Example:
        @use_block_usage_penalty(True)
        class MyHardwarePerfConfig(DefaultHardwarePerfConfig):
            pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls._use_block_usage_penalty = value  # pyre-ignore[16]
        return cls

    return decorator


def use_bytes_for_input_read_size(value: bool = True) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to control whether input_read_size is calculated in bytes or counts.

    When enabled (True, default/OSS behavior):
        input_read_size = batch_inputs * world_size * input_data_type_size  (BYTES)
    When disabled (False, FB legacy behavior):
        input_read_size = batch_inputs * world_size  (COUNT of indices)

    This affects the forward/backward compute formulas where input_read_size is used.

    Args:
        value: Whether to multiply by input_data_type_size (default: True)

    Example:
        @use_bytes_for_input_read_size(False)  # FB legacy: use counts, not bytes
        class GrandTetonHardwarePerfConfig(HardwarePerfConfig):
            pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls._use_bytes_for_input_read_size = value  # pyre-ignore[16]
        return cls

    return decorator


def input_data_type_size(value: float) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to set a custom input_data_type_size for the hardware config.

    This allows hardware configs to override the default BIGINT_DTYPE (8 bytes)
    used in ShardPerfContext. FB legacy estimators use INT_DTYPE (4 bytes).

    Args:
        value: The input data type size in bytes (e.g., 4.0 for INT_DTYPE, 8.0 for BIGINT_DTYPE)

    Example:
        @input_data_type_size(4.0)  # FB legacy uses 4 bytes, not 8
        class GrandTetonHardwarePerfConfig(HardwarePerfConfig):
            pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls._input_data_type_size = value  # pyre-ignore[16]
        return cls

    return decorator


def supported_sharding_types(*sharding_types: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to specify which sharding types a HardwarePerfConfig supports.

    If not specified, all sharding types are evaluated (default behavior).
    If specified, only listed sharding types are evaluated; others raise ValueError.

    The annotation is inherited - child classes get parent's supported types
    unless they override with their own @supported_sharding_types annotation.

    Args:
        *sharding_types: Sharding type values (e.g., "table_wise", "row_wise")

    Example:
        @supported_sharding_types("table_wise", "row_wise")
        class HeterogeneousHardwarePerfConfig(HardwarePerfConfig):
            pass

    Raises:
        ValueError: When attempting to evaluate an unsupported sharding type
    """

    def decorator(cls: Type[T]) -> Type[T]:
        # Store as a frozenset for O(1) lookup, lowercase for case-insensitive matching
        cls._supported_sharding_types = frozenset(  # pyre-ignore[16]
            st.lower() for st in sharding_types
        )
        return cls

    return decorator


# =============================================================================
# SECTION 5: Coefficient Decorators
# =============================================================================
# These decorators allow defining performance coefficients via annotations
# instead of manually creating PerfCoefficientConfig objects. The evaluator
# uses these annotations to discover coefficients for each sharding type.
#
# Usage example:
#     class MyHardwarePerfConfig(HardwarePerfConfig):
#         @fwd_coefficient(sharding_type=["table_wise", "column_wise"])
#         def tw_cw_fwd(self) -> PerfCoefficient:
#             return PerfCoefficient(
#                 input_read_size_multiplier=100.0,
#                 lookup_size_multiplier=1.0,
#                 embedding_output_multiplier=1.0,
#                 hash_size_multiplier=4.5,
#             )
#
#         @bwd_coefficient(sharding_type=["table_wise", "column_wise"])
#         def tw_cw_bwd(self) -> PerfCoefficient:
#             return PerfCoefficient(...)
#
#         @prefetch_coefficient()
#         def prefetch_coeff(self) -> PrefetchCoefficients:
#             return PrefetchCoefficients(...)
# =============================================================================


def fwd_coefficient(
    sharding_type: Optional[Union[str, list[str]]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to mark a method as providing forward pass performance coefficients.

    The decorated method should return a PerfCoefficient object. Method name is
    arbitrary - only the annotation matters for discovery.

    Args:
        sharding_type: Sharding type(s) this coefficient applies to.
                       Can be a single string or list of strings.
                       If None, applies to all sharding types.

    Example:
        @fwd_coefficient(sharding_type=["table_wise", "column_wise"])
        def compute_tw_cw_fwd(self) -> PerfCoefficient:
            return PerfCoefficient(
                input_read_size_multiplier=100.0,
                lookup_size_multiplier=1.0,
                embedding_output_multiplier=1.0,
                hash_size_multiplier=4.5,
            )
    """

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        method._is_fwd_coefficient = True  # pyre-ignore[16]
        method._coefficient_sharding_type = sharding_type  # pyre-ignore[16]
        return method

    return decorator


def bwd_coefficient(
    sharding_type: Optional[Union[str, list[str]]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to mark a method as providing backward pass performance coefficients.

    The decorated method should return a PerfCoefficient object. Method name is
    arbitrary - only the annotation matters for discovery.

    Args:
        sharding_type: Sharding type(s) this coefficient applies to.
                       Can be a single string or list of strings.
                       If None, applies to all sharding types.

    Example:
        @bwd_coefficient(sharding_type=["table_wise", "column_wise"])
        def compute_tw_cw_bwd(self) -> PerfCoefficient:
            return PerfCoefficient(
                input_read_size_multiplier=600.0,
                lookup_size_multiplier=3.0,
                embedding_output_multiplier=3.0,
                hash_size_multiplier=9.0,
            )
    """

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        method._is_bwd_coefficient = True  # pyre-ignore[16]
        method._coefficient_sharding_type = sharding_type  # pyre-ignore[16]
        return method

    return decorator


def prefetch_coefficient() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to mark a method as providing prefetch pipeline coefficients.

    The decorated method should return a PrefetchCoefficients object.
    Prefetch coefficients are global (no sharding type filter).

    Example:
        @prefetch_coefficient()
        def get_prefetch_coeff(self) -> PrefetchCoefficients:
            return PrefetchCoefficients(
                expected_num_lookups_coefficient=4.337620774386766e-07,
                expected_num_unique_lookups_coefficient=1.0654341763287636e-05,
                expected_size_cache_fetches_coefficient=1.3311586664661257e-07,
            )
    """

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        method._is_prefetch_coefficient = True  # pyre-ignore[16]
        return method

    return decorator


# =============================================================================
# Helper Functions for Coefficient Annotations
# =============================================================================


def _matches_coefficient_sharding_type(
    method: Callable[..., Any], sharding_type: str
) -> bool:
    """
    Check if a coefficient method matches the given sharding type.

    Args:
        method: The coefficient method to check
        sharding_type: The sharding type to match against (e.g., 'table_wise')

    Returns:
        True if the method applies to this sharding type
    """
    method_sharding_type = getattr(method, "_coefficient_sharding_type", None)
    if method_sharding_type is None:
        return True  # No filter = matches all

    if isinstance(method_sharding_type, (list, tuple)):
        return sharding_type.lower() in [st.lower() for st in method_sharding_type]

    return method_sharding_type.lower() == sharding_type.lower()


def _get_coefficient_method(
    config: Any,
    marker_attr: str,
    sharding_type: Optional[str] = None,
) -> Optional[Callable[..., Any]]:
    """
    Find a coefficient method on config marked with the given attribute.

    Scans ALL methods by attribute (not by method name). Returns the first
    method that has the marker attribute and matches the sharding type.

    Args:
        config: The hardware config to scan
        marker_attr: The annotation attribute to look for (e.g., '_is_fwd_coefficient')
        sharding_type: Optional sharding type to filter by

    Returns:
        The matching method if found, otherwise None
    """
    # Skip these attributes to avoid infinite recursion (coefficients property
    # calls get_fwd_coefficient which calls this function)
    skip_attrs = {"coefficients", "_coefficients"}

    for attr_name in dir(config):
        if attr_name.startswith("_") or attr_name in skip_attrs:
            continue
        method = getattr(config, attr_name, None)
        if method and callable(method) and getattr(method, marker_attr, False):
            if sharding_type is not None:
                if not _matches_coefficient_sharding_type(method, sharding_type):
                    continue
            return method
    return None


def get_fwd_coefficient(
    config: Any,
    sharding_type: str,
) -> Optional[Any]:
    """
    Get forward coefficient for the given sharding type from config annotations.

    Scans all methods on the config for @fwd_coefficient annotation.
    Returns the PerfCoefficient if a matching method is found.

    Args:
        config: The hardware config to check
        sharding_type: The sharding type to get coefficient for

    Returns:
        PerfCoefficient if annotated method found, otherwise None
    """
    method = _get_coefficient_method(config, "_is_fwd_coefficient", sharding_type)
    if method:
        return method()
    return None


def get_bwd_coefficient(
    config: Any,
    sharding_type: str,
) -> Optional[Any]:
    """
    Get backward coefficient for the given sharding type from config annotations.

    Scans all methods on the config for @bwd_coefficient annotation.
    Returns the PerfCoefficient if a matching method is found.

    Args:
        config: The hardware config to check
        sharding_type: The sharding type to get coefficient for

    Returns:
        PerfCoefficient if annotated method found, otherwise None
    """
    method = _get_coefficient_method(config, "_is_bwd_coefficient", sharding_type)
    if method:
        return method()
    return None


def get_prefetch_coefficient(
    config: Any,
) -> Optional[Any]:
    """
    Get prefetch coefficients from config annotations.

    Scans all methods on the config for @prefetch_coefficient annotation.
    Returns the PrefetchCoefficients if a matching method is found.

    Args:
        config: The hardware config to check

    Returns:
        PrefetchCoefficients if annotated method found, otherwise None
    """
    method = _get_coefficient_method(config, "_is_prefetch_coefficient")
    if method:
        return method()
    return None
