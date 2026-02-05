#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
from itertools import chain
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from torch.distributed._composable.fsdp.fully_shard import FSDPModule as FSDP2
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.fx.immutable_collections import (
    immutable_dict as fx_immutable_dict,
    immutable_list as fx_immutable_list,
)
from torch.fx.node import Node
from torchrec.distributed.logger import static_logger
from torchrec.distributed.model_parallel import ShardedModule
from torchrec.distributed.train_pipeline.pipeline_context import TrainPipelineContext
from torchrec.distributed.train_pipeline.postproc import PipelinedPostproc
from torchrec.distributed.train_pipeline.types import ArgInfo, BaseArgInfoStep, CallArgs

logger: logging.Logger = logging.getLogger(__name__)


class NoopArgInfoStep(BaseArgInfoStep):
    # pyre-ignore
    def process(self, arg) -> Any:
        return arg


class GetAttrArgInfoStep(BaseArgInfoStep):
    def __init__(self, attr_name: str) -> None:
        super().__init__()
        self.attr_name = attr_name

    # pyre-ignore
    def process(self, arg) -> Any:
        return getattr(arg, self.attr_name)


class GetItemArgInfoStep(BaseArgInfoStep):
    def __init__(self, item_index: Union[str, int]) -> None:
        super().__init__()
        self.item_index = item_index

    # pyre-ignore
    def process(self, arg) -> Any:
        return arg[self.item_index]


class PostprocArgInfoStep(BaseArgInfoStep):
    def __init__(self, postproc_module: PipelinedPostproc) -> None:
        super().__init__()
        self.postproc_module = postproc_module

    # pyre-ignore
    def process(self, arg) -> Any:
        return self.postproc_module(arg)


class ScalarArgInfoStep(BaseArgInfoStep):
    def __init__(self, value: object) -> None:
        super().__init__()
        self.value = value

    # pyre-ignore
    def process(self, _arg) -> Any:
        return self.value


class ListArgInfoStep(BaseArgInfoStep):
    def __init__(self, value: List[object]) -> None:
        super().__init__()
        self.value = value

    # pyre-ignore
    def process(self, arg) -> Any:
        return [
            (v if not isinstance(v, ArgInfo) else v.process_steps(arg))
            for v in self.value
        ]


class DictArgInfoStep(BaseArgInfoStep):
    def __init__(self, value: Dict[str, object]) -> None:
        super().__init__()
        self.value = value

    # pyre-ignore
    def process(self, arg) -> Any:
        return {
            k: (v if not isinstance(v, ArgInfo) else v.process_steps(arg))
            for k, v in self.value.items()
        }


class ArgInfoStepFactory:
    """
    Convenience class to reduce the amount of imports the external uses will have.
    Should closely follow the constructor interfaces for the corresponding classes.
    """

    @classmethod
    def noop(cls) -> NoopArgInfoStep:
        return NoopArgInfoStep()

    @classmethod
    def get_attr(cls, name: str) -> GetAttrArgInfoStep:
        return GetAttrArgInfoStep(name)

    @classmethod
    def get_item(cls, index: Union[str, int]) -> GetItemArgInfoStep:
        return GetItemArgInfoStep(index)

    @classmethod
    def postproc(
        cls, pipelined_postproc_module: PipelinedPostproc
    ) -> PostprocArgInfoStep:
        return PostprocArgInfoStep(pipelined_postproc_module)

    @classmethod
    def from_scalar(cls, value: object) -> ScalarArgInfoStep:
        return ScalarArgInfoStep(value)

    @classmethod
    def from_list(cls, value: List[object]) -> ListArgInfoStep:
        return ListArgInfoStep(value)

    @classmethod
    def from_dict(cls, value: Dict[str, object]) -> DictArgInfoStep:
        return DictArgInfoStep(value)


def _check_args_for_call_module(
    node: torch.fx.Node,
) -> bool:
    """
    Recursively checks if args to a node is the result of a call_module.
    """
    if node.op == "call_module":
        return True

    for arg in node.args:
        if isinstance(arg, torch.fx.Node) and _check_args_for_call_module(arg):
            return True

    return False


def _check_postproc_pipelineable(
    module: torch.nn.Module,
) -> bool:
    for _, _ in module.named_parameters(recurse=True):
        # Cannot have any trainable params for it to be pipelined
        logger.warning(
            f"Module {module} cannot be pipelined as it has trainable parameters"
        )
        return False
    return True


def _find_postproc_module_recursive(
    module: torch.nn.Module,
    postproc_module_fqn: str,
) -> Optional[torch.nn.Module]:
    """
    Finds the postproc module in the model.
    """
    for name, child in module.named_modules():
        if name == postproc_module_fqn:
            return child
    return None


class NodeArgsHelper:
    def __init__(
        self,
        model: torch.nn.Module,
        context: TrainPipelineContext,
        pipeline_postproc: bool,
        default_stream: Optional[torch.Stream] = None,
        dist_stream: Optional[torch.Stream] = None,
    ) -> None:
        self._model = model
        self._context = context
        self._pipeline_postproc = pipeline_postproc
        self._default_stream = default_stream
        self._dist_stream = dist_stream
        self._pipelined_postprocs: Set[PipelinedPostproc] = set()

    @property
    def pipelined_postprocs(self) -> Set[PipelinedPostproc]:
        return self._pipelined_postprocs

    def _swap_postproc_module_recursive(
        self,
        module: torch.nn.Module,
        to_swap_module: torch.nn.Module,
        postproc_module_fqn: str,
        path: str = "",
    ) -> torch.nn.Module:
        """
        Swaps the postproc module in the model.
        """
        if isinstance(module, PipelinedPostproc):
            return module

        if path == postproc_module_fqn:
            return to_swap_module

        for name, child in module.named_children():
            child = self._swap_postproc_module_recursive(
                child,
                to_swap_module,
                postproc_module_fqn,
                path + "." + name if path else name,
            )
            setattr(module, name, child)

        return module

    def _handle_constant(
        self,
        arg: Any,
        arg_info: ArgInfo,
        for_postproc_module: bool = False,
    ) -> Optional[ArgInfo]:
        if not self._pipeline_postproc:
            return None

        if isinstance(arg, fx_immutable_dict):
            step = ArgInfoStepFactory.from_dict(
                {
                    k: self._handle_collection_element(v, for_postproc_module)
                    for k, v in arg.items()
                }
            )
        elif isinstance(arg, fx_immutable_list):
            step = ArgInfoStepFactory.from_list(
                [self._handle_collection_element(v, for_postproc_module) for v in arg]
            )
        else:
            step = ArgInfoStepFactory.from_scalar(arg)
        arg_info.add_step(step)
        return arg_info

    def _handle_collection_element(
        self,
        arg: Any,
        for_postproc_module: bool = False,
    ) -> Any:
        if not isinstance(arg, torch.fx.Node):
            return arg

        arg_info_nested = self._get_node_args_helper_inner(
            arg,
            for_postproc_module,
        )
        return arg_info_nested

    def _handle_placeholder(
        self, child_node: torch.fx.Node, arg_info: ArgInfo
    ) -> ArgInfo:
        # note: mutates arg_info
        if hasattr(child_node, "ph_key"):
            # pyre-fixme[16]
            ph_key: str = child_node.ph_key
            # example: ph_key = 'event_id_list_features_seqs[marketplace]'
            ph_key = ph_key.replace("[", ".")
            ph_keys = ph_key.split(".")
            for key in ph_keys:
                if "]" in key:
                    k_ = key[:-1]
                    try:
                        k_ = int(k_)
                    except ValueError:
                        pass
                    arg_info.append_step(ArgInfoStepFactory.get_item(k_))
                else:
                    arg_info.append_step(ArgInfoStepFactory.get_attr(key))
        else:
            # no-op
            arg_info.add_step(ArgInfoStepFactory.noop())
        return arg_info

    def _handle_module(
        self, child_node: torch.fx.Node, arg_info: ArgInfo
    ) -> Optional[ArgInfo]:
        postproc_module_fqn = str(child_node.target)
        postproc_module = _find_postproc_module_recursive(
            self._model, postproc_module_fqn
        )

        if not self._pipeline_postproc:
            logger.warning(
                f"Found module {postproc_module} that potentially modifies input KJT. "
                "Train pipeline initialized with `pipeline_postproc=False` (default), "
                "so we assume KJT input modification. "
                "To allow torchrec to check if this module can be safely pipelined, "
                "please set `pipeline_postproc=True`"
            )
            return None

        if not postproc_module:
            # Could not find such module, should not happen
            return None

        if isinstance(postproc_module, PipelinedPostproc):
            # Already did module swap and registered args, early exit
            self._pipelined_postprocs.add(postproc_module)
            arg_info.add_step(ArgInfoStepFactory.postproc(postproc_module))
            return arg_info

        if not isinstance(postproc_module, torch.nn.Module):
            logger.warning(
                f"Expected postproc_module to be nn.Module but was {type(postproc_module)}"
            )
            return None

        # check if module is safe to pipeline i.e.no trainable param
        if not _check_postproc_pipelineable(postproc_module):
            return None

        # For module calls, `self` isn't counted
        total_num_args = len(child_node.args) + len(child_node.kwargs)
        if total_num_args == 0:
            # module call without any args, assume KJT modified
            return None

        # recursive call to check that all inputs to this postproc module
        # is either made of postproc module or non-modifying train batch input
        # transformations
        postproc_args, num_found_safe_postproc_args = self.get_node_args(
            child_node,
            for_postproc_module=True,
        )
        if num_found_safe_postproc_args == total_num_args:
            logger.info(
                f"Module {postproc_module} is a valid postproc module (no "
                "trainable params and inputs can be derived from train batch input "
                "via a series of either valid postproc modules or non-modifying "
                "transformations) and will be applied during sparse data dist stage"
            )

            pipelined_postproc_module = PipelinedPostproc(
                postproc_module,
                postproc_module_fqn,
                postproc_args,
                self._context,
                default_stream=self._default_stream,
                dist_stream=self._dist_stream,
            )

            # module swap
            self._model = self._swap_postproc_module_recursive(
                self._model, pipelined_postproc_module, postproc_module_fqn
            )

            self._pipelined_postprocs.add(pipelined_postproc_module)
            arg_info.add_step(ArgInfoStepFactory.postproc(pipelined_postproc_module))
            return arg_info

        return None

    def _get_node_args_helper_inner(
        self,
        # pyre-ignore
        arg,
        for_postproc_module: bool = False,
    ) -> Optional[ArgInfo]:
        arg_info = ArgInfo([])
        while True:
            if not isinstance(arg, torch.fx.Node):
                return self._handle_constant(arg, arg_info, for_postproc_module)

            child_node = arg

            if child_node.op == "placeholder":
                return self._handle_placeholder(arg, arg_info)
            elif child_node.op == "call_module":
                return self._handle_module(arg, arg_info)
            elif (
                child_node.op == "call_function"
                and child_node.target.__module__ == "builtins"
                # pyre-fixme[16]
                and child_node.target.__name__ == "getattr"
            ):
                arg_info.add_step(
                    # pyre-fixme[6]: For 2nd argument expected `str` but got Unknown
                    ArgInfoStepFactory.get_attr(child_node.args[1])
                )
                arg = child_node.args[0]
            elif (
                child_node.op == "call_function"
                and child_node.target.__module__ == "_operator"
                # pyre-fixme[16]
                and child_node.target.__name__ == "getitem"
            ):
                arg_info.add_step(
                    # pyre-fixme[6]: For 2nd argument expected `str` but got Unknown
                    ArgInfoStepFactory.get_item(child_node.args[1])
                )
                arg = child_node.args[0]
            elif (
                child_node.op == "call_function"
                and child_node.target.__module__ == "torch.utils._pytree"
                # pyre-fixme[16]
                and child_node.target.__name__ == "tree_unflatten"
            ):
                """
                This is for the PT2 export path where we unflatten the input to reconstruct
                the structure with the recorded tree spec.
                """
                step = arg_info.steps[0]
                assert isinstance(step, GetItemArgInfoStep)
                # pyre-fixme[16]
                arg = child_node.args[0][step.item_index]
            elif (
                child_node.op == "call_function"
                and child_node.target.__module__ == "torchrec.sparse.jagged_tensor"
                # pyre-fixme[16]
                and child_node.target.__name__ == "KeyedJaggedTensor"
            ):
                call_module_found = False

                for arg_node in chain(child_node.args, child_node.kwargs.values()):
                    if isinstance(
                        arg_node, torch.fx.Node
                    ) and _check_args_for_call_module(arg_node):
                        call_module_found = True
                        break

                if call_module_found:
                    break

                if "values" in child_node.kwargs:
                    arg = child_node.kwargs["values"]
                else:
                    arg = child_node.args[1]

            elif child_node.op == "call_method" and child_node.target == "get":
                # pyre-ignore[6]
                arg_info.add_step(ArgInfoStepFactory.get_item(child_node.args[1]))
                arg = child_node.args[0]
            else:
                logger.warning(
                    f"fx node {child_node.name, child_node.op, child_node.target} "
                    "can't be handled correctly for postproc module"
                )
                break

        # if we couldn't hit one of the "decisive" outcomes (constant, placeholder or module), return "not found"
        return None

    def _get_node_args_helper(
        self,
        arguments,  # pyre-ignore[2]
        # Add `None` constants to arg info only for postproc modules
        # Defaults to False for backward compatibility
        for_postproc_module: bool = False,
    ) -> Tuple[List[ArgInfo], int]:
        """
        Goes through the args/kwargs of a node and arranges them into a list of `ArgInfo`s.
        It also counts the number of (args + kwargs) found.
        """
        num_found = 0
        arg_info_list = []
        for arg in arguments:
            if not for_postproc_module and arg is None:
                arg_info = ArgInfo([ArgInfoStepFactory.from_scalar(None)])
                arg_info_list.append(arg_info)
                num_found += 1
                continue
            arg_info = self._get_node_args_helper_inner(
                arg,
                for_postproc_module,
            )
            if arg_info is not None:
                num_found += 1
                arg_info_list.append(arg_info)
        return arg_info_list, num_found

    def get_node_args(
        self,
        node: Node,
        for_postproc_module: bool = False,
    ) -> Tuple[CallArgs, int]:
        pos_arg_info_list, args_found = self._get_node_args_helper(
            node.args,
            for_postproc_module,
        )
        kwargs_arg_info_list, kwargs_found = self._get_node_args_helper(
            node.kwargs.values(),
            for_postproc_module,
        )

        # Replace with proper names for kwargs
        kwargs_info_list = dict(zip(node.kwargs, kwargs_arg_info_list))

        return CallArgs(pos_arg_info_list, kwargs_info_list), args_found + kwargs_found


def _get_leaf_module_names(model: torch.nn.Module) -> List[str]:
    """
    Returns a list of top level modules to be used as leaf modules for FX tracing.
    This is a shallow FX trace that only goes the minimum depth required to pipeline.
    Any sub-module who does not contain a ShardedModule would be considered as a leaf
    module unless explicitly tagged as `_is_pytorch_fx_traceable = True`.

    disclaimer:
        the algorithm is based on "named_modules()" API from torch.nn.Module, there
    could be corner cases that a returned "leaf_module" is not actually called in the
    forward pass. for example the umbrella_module would be considered as the top-level
    leaf module but it actually won't appear in a fx-traced graph.

    ```
        # the main_model's hierarchy looks like below:
        main_model
            - sharded_module
            - umbrella_module
                - actual_leaf_module_1
                - actual_leaf_module_2

        # and the main_model's forward is something like:
        def forward(self, x1, x2, x3):
            emb1 = self.sharded_module(x1)
            emb2 = self.umbrella_module.actual_leaf_module_1(x2)
            emb3 = self.umbrella_module.actual_leaf_module_2(x3)
            return emb1 + emb2 + emb3
    ```

    """

    def _get_leaf_module_names_helper(
        model: torch.nn.Module,
        path: str,
        leaf_module_names: Set[str],
    ) -> bool:
        """
        recursive function returns True if any of the sub-modules is ShardedModule.
        it also added the fqns of the sub-modules who do not contain any ShardedModule
        into the `leaf_module_names` unless it's marked as `_is_pytorch_fx_traceable = True`,
        which suggests this ShardedModule-free module should NOT be treated as a leaf module
        """
        sharded_children = set()
        for name, child in model.named_children():
            curr_path = path + name
            if isinstance(child, ShardedModule):
                sharded_children.add(name)
            else:
                child_sharded = _get_leaf_module_names_helper(
                    child,
                    curr_path + ".",
                    leaf_module_names,
                )
                if child_sharded:
                    sharded_children.add(name)

        # only do this for hybrid module (has sharded child)
        if len(sharded_children) > 0:
            for name, child in model.named_children():
                if name in sharded_children:
                    continue
                # assume module is leaf node unless annotated otherwise
                if not getattr(child, "_is_pytorch_fx_traceable", False):
                    leaf_module_names.add(path + name)
        return len(sharded_children) > 0

    leaf_module_names: Set[str] = set()
    _get_leaf_module_names_helper(
        model,
        "",
        leaf_module_names,
    )
    return list(leaf_module_names)


class Tracer(torch.fx.Tracer):
    """
    The Trace class used in `_rewrite_model`, treating all ShardedModules and ShardedModule-free
    modules as leaf modules. A module who is not a ShardedModule but contains ShardedModule would
    NOT be considered as a leaf module.
    """

    # Disables proxying buffers during tracing. Ideally, proxying buffers would be
    # disabled, but some models are currently mutating buffer values, which causes errors
    # during tracing. If those models can be rewritten to not do that, we can likely
    # remove this line.
    proxy_buffer_attributes = False

    def __init__(
        self, leaf_modules: Optional[List[str]] = None, extend_leaf_fqn: bool = False
    ) -> None:
        """
        Initializes the Tracer for FX tracing with custom leaf module handling.

        Args:
            leaf_modules: Optional list of fully qualified names (FQNs) of modules to treat
                as leaf modules during tracing. If None, defaults to an empty list.
            extend_leaf_fqn: If True, treats any module whose FQN starts with a leaf module
                FQN as a leaf module (includes submodules). If False, only exact matches
                are considered leaf modules. Defaults to False.
        """
        super().__init__()
        self._leaf_modules: List[str] = leaf_modules if leaf_modules is not None else []
        self._extend_leaf_fqn = extend_leaf_fqn
        if self._extend_leaf_fqn:
            static_logger.info("use extended leaf module fqn in pipeline tracer")

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if (
            isinstance(m, ShardedModule)
            or module_qualified_name in self._leaf_modules
            or isinstance(m, FSDP)
            or isinstance(m, FSDP2)
        ):
            return True
        if self._extend_leaf_fqn:
            if self.is_extended_leaf_modules(m, module_qualified_name):
                return True
        return super().is_leaf_module(m, module_qualified_name)

    def is_extended_leaf_modules(
        self, m: torch.nn.Module, module_qualified_name: str
    ) -> bool:
        for leaf_module in self._leaf_modules:
            if module_qualified_name.startswith(leaf_module):
                # in a corner case that the fqn == 'main_model.leaf_module.submod'
                # we should consider this fqn also a leaf_module
                if (
                    module_qualified_name != leaf_module
                    and module_qualified_name[len(leaf_module)] == "."
                ):
                    static_logger.warning(
                        f"extended leaf modules: module '{module_qualified_name}'"
                        f" is a sub-module of '{leaf_module}'"
                    )
                    return True
        return False
