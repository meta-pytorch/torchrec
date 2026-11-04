#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
import os
import signal
import tempfile
import threading
import time
import unittest
from typing import Any, Optional, TypeVar
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torchrec.metrics.cpu_offloaded_metric_ipc import (
    build_slot_layout,
    SharedUpdateRing,
)
from torchrec.metrics.cpu_offloaded_metric_module import CPUOffloadedRecMetricModule
from torchrec.metrics.cpu_offloaded_metric_process import (
    _MIN_RING_SLOTS,
    _RingWriter,
    _WorkerFuture,
    ProcessIsolatedCPUOffloadedRecMetricModule,
)
from torchrec.metrics.cpu_offloaded_metric_worker import (
    _MetricWorker,
    Control,
    ControlMessage,
    oss_worker_factory,
    require_handlers,
)
from torchrec.metrics.metric_job_types import MetricUpdateJob
from torchrec.metrics.metric_module import generate_metric_module, RecMetricModule
from torchrec.metrics.metrics_config import (
    DefaultMetricsConfig,
    MetricsConfig,
    RecMetricDef,
    RecMetricEnum,
    ThroughputDef,
)
from torchrec.metrics.rec_metric import RecMetricException, RecMetricValidationError
from torchrec.metrics.test_utils import gen_test_batch, gen_test_tasks
from torchrec.test_utils import init_process_group_single_rank

BATCH_SIZE = 64
_ModuleT = TypeVar("_ModuleT", bound=RecMetricModule)
TASKS = gen_test_tasks(["t1", "t2"])


_DEFAULT_METRICS: tuple[RecMetricEnum, ...] = (
    RecMetricEnum.NE,
    RecMetricEnum.CTR,
    RecMetricEnum.CALIBRATION,
    RecMetricEnum.MSE,
    RecMetricEnum.AUC,
)
_EXTENDED_METRICS: tuple[RecMetricEnum, ...] = _DEFAULT_METRICS + (
    RecMetricEnum.ACCURACY,
    RecMetricEnum.AUPRC,
    RecMetricEnum.MAE,
    RecMetricEnum.NE_POSITIVE,
    RecMetricEnum.UNWEIGHTED_NE,
    RecMetricEnum.CALI_FREE_NE,
    RecMetricEnum.PRECISION,
    RecMetricEnum.RECALL,
    RecMetricEnum.NUM_POSITIVE_SAMPLES,
    RecMetricEnum.SUM_WEIGHTS,
    RecMetricEnum.WEIGHTED_SUM_PREDICTIONS,
)


def _metrics_config(
    window_size: int = 10_000,
    throughput: bool = False,
    metrics: tuple[RecMetricEnum, ...] = _DEFAULT_METRICS,
) -> MetricsConfig:
    return dataclasses.replace(
        DefaultMetricsConfig,
        rec_tasks=TASKS,
        rec_metrics={
            metric: RecMetricDef(rec_tasks=TASKS, window_size=window_size)
            for metric in metrics
        },
        throughput_metric=ThroughputDef() if throughput else None,
        state_metrics=[],
    )


def _batch(seed: int, batch_size: int = BATCH_SIZE) -> dict[str, torch.Tensor]:
    model_out: dict[str, torch.Tensor] = {}
    for index, task in enumerate(TASKS):
        batch = gen_test_batch(
            batch_size,
            label_name=task.label_name,
            prediction_name=task.prediction_name,
            weight_name=task.weight_name,
            seed=seed * 10 + index,
        )
        model_out.update(
            {
                task.label_name: batch[task.label_name],
                task.prediction_name: batch[task.prediction_name],
                task.weight_name: batch[task.weight_name],
            }
        )
    return model_out


def _threaded_module(config: MetricsConfig) -> RecMetricModule:
    return generate_metric_module(
        metric_class=CPUOffloadedRecMetricModule,
        metrics_config=config,
        batch_size=BATCH_SIZE,
        world_size=1,
        my_rank=0,
        state_metrics_mapping={},
        device=torch.device("cpu"),
        module_kwargs={"model_out_device": torch.device("cpu"), "update_batch_size": 1},
    )


class _KwargProbeModule(CPUOffloadedRecMetricModule):
    """Reports a checksum of the update kwargs it received as a metric."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.kwarg_probe = 0.0

    def _process_metric_update_job(self, metric_update_job: MetricUpdateJob) -> None:
        kwargs = metric_update_job.kwargs
        self.kwarg_probe += (
            float(kwargs["probe"].sum())
            + float(kwargs["nested"]["t"].sum())
            + kwargs["scale"]
            + kwargs["nested"]["s"]
        )
        metric_update_job.kwargs = {}
        super()._process_metric_update_job(metric_update_job)

    def _process_metric_compute_job(self, metric_compute_job: Any) -> Any:
        result = super()._process_metric_compute_job(metric_compute_job)
        result["kwarg_probe"] = torch.tensor(self.kwarg_probe, dtype=torch.double)
        return result


class _MergeProbeModule(CPUOffloadedRecMetricModule):
    """Reports the largest number of updates merged into one job."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_merged = 0

    def _process_metric_update_job(self, metric_update_job: MetricUpdateJob) -> None:
        self.max_merged = max(self.max_merged, metric_update_job.merged_count)
        super()._process_metric_update_job(metric_update_job)

    def _process_metric_compute_job(self, metric_compute_job: Any) -> Any:
        result = super()._process_metric_compute_job(metric_compute_job)
        result["max_merged"] = torch.tensor(self.max_merged)
        return result


class _SlowFailingUpdateModule(CPUOffloadedRecMetricModule):
    def _process_metric_update_job(self, metric_update_job: MetricUpdateJob) -> None:
        time.sleep(1.0)
        raise RuntimeError("injected update failure")


def _spawn_module_and_exit(conn: Any) -> None:
    module = _process_module(_metrics_config())
    conn.send(module._process.pid)
    conn.close()
    os._exit(0)


def _two_rank_modules(
    rank: int, config: MetricsConfig
) -> tuple[RecMetricModule, RecMetricModule]:
    cpu = torch.device("cpu")
    common: dict[str, Any] = {
        "metrics_config": config,
        "batch_size": BATCH_SIZE,
        "world_size": 2,
        "my_rank": rank,
        "state_metrics_mapping": {},
        "device": cpu,
    }
    isolated = generate_metric_module(
        metric_class=ProcessIsolatedCPUOffloadedRecMetricModule,
        module_kwargs={
            "model_out_device": cpu,
            "worker_factory": oss_worker_factory(config, BATCH_SIZE, 2, rank),
        },
        **common,
    )
    threaded = generate_metric_module(
        metric_class=CPUOffloadedRecMetricModule,
        module_kwargs={"model_out_device": cpu, "update_batch_size": 1},
        **common,
    )
    return isolated, threaded


def _assert_close(actual: Any, expected: Any) -> None:
    assert sorted(actual) == sorted(expected), (sorted(actual), sorted(expected))
    for key, value in expected.items():
        torch.testing.assert_close(
            torch.as_tensor(actual[key]).double(),
            torch.as_tensor(value).double(),
            msg=key,
            equal_nan=True,
        )


def _two_rank_trainer(rank: int, store_path: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=2
    )
    config = _metrics_config()
    isolated, threaded = _two_rank_modules(rank, config)
    for step in range(6):
        batch = _batch(2 * step + rank)
        isolated.update(batch)
        threaded.update(batch)
    _assert_close(
        isolated.async_compute().resolve(), threaded.async_compute().resolve()
    )
    states = (isolated.state_dict(), threaded.state_dict())
    _assert_close(*states)
    isolated.shutdown()
    threaded.shutdown()

    restored = _two_rank_modules(rank, config)
    for module, state in zip(restored, states):
        module.load_state_dict(state)
    for step in range(6, 9):
        batch = _batch(2 * step + rank)
        for module in restored:
            module.update(batch)
    _assert_close(*(module.async_compute().resolve() for module in restored))
    for module in restored:
        module.shutdown()
    dist.destroy_process_group()


def _process_gone(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(") ", 1)[1].startswith("Z")
    except FileNotFoundError:
        return True


def _process_module(
    config: MetricsConfig,
    worker_module_kwargs: Optional[dict[str, Any]] = None,
    metric_class: type[CPUOffloadedRecMetricModule] = CPUOffloadedRecMetricModule,
    **kwargs: Any,
) -> ProcessIsolatedCPUOffloadedRecMetricModule:
    module = generate_metric_module(
        metric_class=ProcessIsolatedCPUOffloadedRecMetricModule,
        metrics_config=config,
        batch_size=BATCH_SIZE,
        world_size=1,
        my_rank=0,
        state_metrics_mapping={},
        device=torch.device("cpu"),
        module_kwargs={
            "model_out_device": torch.device("cpu"),
            "worker_factory": oss_worker_factory(
                config, BATCH_SIZE, 1, 0, worker_module_kwargs, metric_class
            ),
            **kwargs,
        },
    )
    assert isinstance(module, ProcessIsolatedCPUOffloadedRecMetricModule)
    return module


def _assert_metrics_equal(
    test: unittest.TestCase, actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    test.assertEqual(sorted(actual), sorted(expected))
    for key, value in expected.items():
        torch.testing.assert_close(
            torch.as_tensor(actual[key]).double(),
            torch.as_tensor(value).double(),
            msg=key,
            equal_nan=True,
        )


class ProcessIsolatedCPUOffloadedRecMetricModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        init_process_group_single_rank()
        self._modules: list[RecMetricModule] = []

    def tearDown(self) -> None:
        for module in self._modules:
            module.shutdown()
        dist.destroy_process_group()

    def _track(self, module: _ModuleT) -> _ModuleT:
        self._modules.append(module)
        return module

    def test_computes_match_threaded_module_at_each_boundary(self) -> None:
        config = _metrics_config(window_size=5 * BATCH_SIZE)
        threaded = self._track(_threaded_module(config))
        isolated = self._track(_process_module(config, num_slots=4))
        threaded_results, isolated_results = [], []
        for step in range(23):
            batch = _batch(step)
            threaded.update(batch)
            isolated.update(batch)
            if step % 7 == 6:
                threaded_results.append(threaded.async_compute())
                isolated_results.append(isolated.async_compute())
        for actual, expected in zip(isolated_results, threaded_results):
            _assert_metrics_equal(self, actual.resolve(), expected.resolve())
        self.assertEqual(len(isolated_results), 3)

    def test_state_dict_matches_threaded_module_and_round_trips(self) -> None:
        config = _metrics_config()
        threaded = self._track(_threaded_module(config))
        isolated = self._track(_process_module(config))
        for step in range(9):
            threaded.update(_batch(step))
            isolated.update(_batch(step))
        threaded.sync()
        isolated.sync()
        expected_state = threaded.state_dict()
        actual_state = isolated.state_dict()
        self.assertEqual(list(actual_state), list(expected_state))
        for key, value in expected_state.items():
            torch.testing.assert_close(actual_state[key], value, msg=key)

        restored = self._track(_process_module(config))
        restored.reset()
        restored.load_state_dict(copy.deepcopy(expected_state))
        reference = self._track(_threaded_module(config))
        reference.reset()
        reference.load_state_dict(copy.deepcopy(expected_state))
        for step in range(9, 12):
            restored.update(_batch(step))
            reference.update(_batch(step))
        _assert_metrics_equal(
            self,
            restored.async_compute().resolve(),
            reference.async_compute().resolve(),
        )

    def test_subscribe_receives_metrics_off_the_caller_thread(self) -> None:
        isolated = self._track(_process_module(_metrics_config()))
        isolated.update(_batch(0))
        received: list[dict[str, Any]] = []
        callback_threads: list[threading.Thread] = []
        callback_done = threading.Event()

        def receive(metrics: dict[str, Any]) -> None:
            received.append(metrics)
            callback_threads.append(threading.current_thread())
            callback_done.set()

        # Hold the request until subscribed, or a fast reply runs the callback inline.
        with patch.object(isolated, "_send_control") as send_control:
            deferrable = isolated.async_compute()
        deferrable.subscribe(receive)
        isolated._send_control(
            *send_control.call_args.args, **send_control.call_args.kwargs
        )
        deferrable.resolve()
        self.assertTrue(callback_done.wait(timeout=120))
        self.assertEqual(callback_threads, [isolated._listener])
        self.assertEqual(len(received), 1)
        self.assertIn("ne-t1|lifetime_ne", received[0])

    def test_compute_is_async_and_throughput_uses_trainer_timestamps(self) -> None:
        isolated = self._track(_process_module(_metrics_config(throughput=True)))
        for step in range(5):
            isolated.update(_batch(step))
        throughput = isolated.compute_throughput().resolve()
        self.assertTrue(any("throughput" in key for key in throughput))
        self.assertTrue(isolated.compute().resolve())

    def test_layout_change_mid_stream(self) -> None:
        config = _metrics_config()
        threaded = self._track(_threaded_module(config))
        isolated = self._track(_process_module(config))
        for step, batch_size in enumerate([BATCH_SIZE, BATCH_SIZE // 2, BATCH_SIZE]):
            threaded.update(_batch(step, batch_size))
            isolated.update(_batch(step, batch_size))
        _assert_metrics_equal(
            self,
            isolated.async_compute().resolve(),
            threaded.async_compute().resolve(),
        )

    def test_ring_grows_when_outputs_outgrow_slots(self) -> None:
        config = _metrics_config()
        threaded = self._track(_threaded_module(config))
        isolated = self._track(_process_module(config, num_slots=8, slot_bytes=1024))
        for step, batch_size in enumerate(
            [BATCH_SIZE // 2, BATCH_SIZE, BATCH_SIZE, BATCH_SIZE // 2] * 4
        ):
            threaded.update(_batch(step, batch_size))
            isolated.update(_batch(step, batch_size))
        _assert_metrics_equal(
            self,
            isolated.async_compute().resolve(),
            threaded.async_compute().resolve(),
        )
        self.assertGreater(isolated._writer.ring.slot_bytes, 1024)
        isolated._writer.release_retired()
        self.assertEqual(isolated._writer._retired, [])

    def test_ring_growth_honors_budget_above_slot_floor(self) -> None:
        budget = 16 << 20
        for payload_bytes, expected_slots in (
            (256 << 10, None),
            (2 << 20, _MIN_RING_SLOTS),
        ):
            writer = _RingWriter(
                SharedUpdateRing(4, 1024),
                torch.device("cpu"),
                ring_bytes_budget=budget,
                max_slots=256,
                on_error=MagicMock(),
            )
            layout = build_slot_layout(
                0, [("x", torch.zeros(payload_bytes, dtype=torch.uint8))]
            )
            ring = writer.maybe_grow(0, layout)
            assert ring is not None
            self.assertEqual(
                ring.num_slots, expected_slots or budget // ring.slot_bytes
            )

    def test_ring_growth_past_hard_cap_fails_loudly(self) -> None:
        writer = _RingWriter(
            SharedUpdateRing(4, 1024),
            torch.device("cpu"),
            ring_bytes_budget=1 << 20,
            max_slots=256,
            on_error=MagicMock(),
        )
        layout = build_slot_layout(0, [("x", torch.zeros(1 << 20, dtype=torch.uint8))])
        with patch(
            "torchrec.metrics.cpu_offloaded_metric_process._MAX_RING_BYTES", 4 << 20
        ), self.assertRaisesRegex(RecMetricException, "would pin more than"):
            writer.maybe_grow(0, layout)

    def test_layout_cache_stays_bounded_under_varying_shapes(self) -> None:
        config = _metrics_config()
        threaded = self._track(_threaded_module(config))
        with patch(
            "torchrec.metrics.cpu_offloaded_metric_process._LAYOUT_CACHE_SIZE", 2
        ):
            isolated = self._track(_process_module(config))
            for step in range(12):
                batch = _batch(step, BATCH_SIZE - step % 4)
                threaded.update(batch)
                isolated.update(batch)
            self.assertEqual(len(isolated._writer._layouts), 2)
        _assert_metrics_equal(
            self,
            isolated.async_compute().resolve(),
            threaded.async_compute().resolve(),
        )

    def test_auto_tuned_compute_interval_reaches_trainer(self) -> None:
        config = dataclasses.replace(
            _metrics_config(), compute_interval_steps=2, min_compute_interval=100.0
        )
        isolated = self._track(_process_module(config))
        for step in range(4):
            isolated.update(_batch(step))
            if isolated.should_compute():
                isolated.async_compute().resolve()
        self.assertGreater(isolated.compute_interval_steps, 2)

    def test_debug_mode_rejects_non_finite_predictions(self) -> None:
        isolated = self._track(_process_module(_metrics_config()))
        isolated._configure_debug_mode(True, 0)
        batch = _batch(0)
        batch[TASKS[0].prediction_name][0] = float("inf")
        with self.assertRaises(RecMetricValidationError):
            isolated.update(batch)

    def test_restart_without_load_state_dict_warns(self) -> None:
        logger_name = "torchrec.metrics.cpu_offloaded_metric_process"
        with patch.dict(os.environ, {"MAST_HPC_JOB_ATTEMPT_INDEX": "1"}):
            isolated = self._track(_process_module(_metrics_config()))
            with self.assertLogs(logger_name, level="WARNING"):
                isolated.update(_batch(0))
            restored = self._track(_process_module(_metrics_config()))
            restored.load_state_dict(isolated.state_dict())
            with self.assertNoLogs(logger_name, level="WARNING"):
                restored.update(_batch(0))
            reset = self._track(_process_module(_metrics_config()))
            reset.reset()
            with self.assertNoLogs(logger_name, level="WARNING"):
                reset.update(_batch(0))

    def test_worker_failure_surfaces_on_trainer(self) -> None:
        isolated = self._track(
            _process_module(
                _metrics_config(), worker_module_kwargs={"not_an_argument": 1}
            )
        )
        with self.assertRaisesRegex(RecMetricException, "metric worker crashed"):
            for step in range(10_000):
                isolated.update(_batch(step % 4))
                isolated.async_compute().resolve()
        isolated.shutdown()

    def test_worker_applies_each_update_individually(self) -> None:
        isolated = self._track(
            _process_module(
                _metrics_config(),
                worker_module_kwargs={"update_batch_size": 10},
                metric_class=_MergeProbeModule,
            )
        )
        for step in range(25):
            isolated.update(_batch(step))
        self.assertEqual(int(isolated.async_compute().resolve()["max_merged"]), 1)

    def test_update_failure_after_compute_is_queued_fails_the_compute(self) -> None:
        isolated = self._track(
            _process_module(_metrics_config(), metric_class=_SlowFailingUpdateModule)
        )
        for step in range(12):
            isolated.update(_batch(step))
        with self.assertRaises(RecMetricException):
            isolated.async_compute().resolve()

    def test_shutdown_is_idempotent_and_stops_worker(self) -> None:
        isolated = _process_module(_metrics_config())
        isolated.update(_batch(0))
        isolated.shutdown()
        isolated.shutdown()
        self.assertFalse(isolated._process.is_alive())
        self.assertEqual(isolated._process.exitcode, 0)
        with self.assertRaises(RecMetricException):
            isolated.update(_batch(1))

    def test_concurrent_shutdown_runs_once(self) -> None:
        isolated = _process_module(_metrics_config())
        isolated.update(_batch(0))
        with patch.object(
            isolated, "_stop_process", wraps=isolated._stop_process
        ) as stop:
            threads = [threading.Thread(target=isolated.shutdown) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        stop.assert_called_once()
        self.assertFalse(isolated._process.is_alive())

    def test_extended_metric_types_match_threaded_module(self) -> None:
        config = _metrics_config(window_size=5 * BATCH_SIZE, metrics=_EXTENDED_METRICS)
        threaded = self._track(_threaded_module(config))
        isolated = self._track(_process_module(config))
        threaded_results, isolated_results = [], []
        for step in range(15):
            threaded.update(_batch(step))
            isolated.update(_batch(step))
            if step % 5 == 4:
                threaded_results.append(threaded.async_compute())
                isolated_results.append(isolated.async_compute())
        for actual, expected in zip(isolated_results, threaded_results):
            resolved = expected.resolve()
            self.assertGreater(len(resolved), 40)
            _assert_metrics_equal(self, actual.resolve(), resolved)

    def test_subscribe_callback_can_issue_control_requests(self) -> None:
        isolated = self._track(_process_module(_metrics_config()))
        isolated.update(_batch(0))
        states: list[dict[str, Any]] = []
        deferrable = isolated.async_compute()
        deferrable.subscribe(lambda _: states.append(isolated.state_dict()))
        waiter = threading.Thread(target=deferrable.resolve, daemon=True)
        waiter.start()
        deadline = time.monotonic() + 120
        while not states and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(len(states), 1)
        self.assertGreater(len(states[0]), 0)

    def test_worker_kill_fails_pending_and_later_calls(self) -> None:
        isolated = self._track(_process_module(_metrics_config()))
        isolated.update(_batch(0))
        isolated.async_compute().resolve()
        pid = isolated._process.pid
        assert pid is not None
        os.kill(pid, signal.SIGKILL)
        with self.assertRaisesRegex(RecMetricException, "exited unexpectedly"):
            for step in range(1, 10_000):
                isolated.update(_batch(step % 4))
                isolated.async_compute().resolve()

    def test_two_ranks_match_threaded_and_restore_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(
                _two_rank_trainer,
                args=(os.path.join(tmp, "store"),),
                nprocs=2,
                join=True,
            )

    def test_worker_exits_when_trainer_process_dies(self) -> None:
        ctx = torch.multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        helper = ctx.Process(target=_spawn_module_and_exit, args=(child_conn,))
        helper.start()
        self.assertTrue(parent_conn.poll(300))
        worker_pid = parent_conn.recv()
        helper.join(timeout=60)
        deadline = time.monotonic() + 60
        while not _process_gone(worker_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertTrue(_process_gone(worker_pid))

    def test_compute_backlog_backpressures_instead_of_failing(self) -> None:
        config = _metrics_config()
        threaded = self._track(_threaded_module(config))
        isolated = self._track(
            _process_module(config, worker_module_kwargs={"compute_queue_size": 2})
        )
        pending = []
        for step in range(30):
            threaded.update(_batch(step))
            isolated.update(_batch(step))
            pending.append(isolated.async_compute())
        results = [deferrable.resolve() for deferrable in pending]
        _assert_metrics_equal(self, results[-1], threaded.async_compute().resolve())

    def test_tensor_and_static_update_kwargs_reach_worker(self) -> None:
        config = _metrics_config()
        module_kwargs = {"update_batch_size": 1}
        threaded = self._track(
            generate_metric_module(
                metric_class=_KwargProbeModule,
                metrics_config=config,
                batch_size=BATCH_SIZE,
                world_size=1,
                my_rank=0,
                state_metrics_mapping={},
                device=torch.device("cpu"),
                module_kwargs={
                    "model_out_device": torch.device("cpu"),
                    **module_kwargs,
                },
            )
        )
        isolated = self._track(
            _process_module(
                config,
                worker_module_kwargs=module_kwargs,
                metric_class=_KwargProbeModule,
            )
        )
        for step in range(9):
            kwargs = {
                "probe": torch.full((3,), float(step)),
                "scale": float(step // 3),
                "nested": {"t": torch.arange(4.0) * step, "s": 10.0 * (step // 3)},
            }
            threaded.update(_batch(step), **kwargs)
            isolated.update(_batch(step), **kwargs)
        actual = isolated.async_compute().resolve()
        expected = threaded.async_compute().resolve()
        self.assertGreater(float(expected["kwarg_probe"]), 0.0)
        _assert_metrics_equal(self, actual, expected)

    def test_recursive_load_through_parent_module(self) -> None:
        config = _metrics_config()
        source = self._track(_threaded_module(config))
        for step in range(6):
            source.update(_batch(step))
        source.sync()
        state = source.state_dict()

        isolated = self._track(_process_module(config))
        parent = torch.nn.Module()
        parent.metrics = isolated
        parent.load_state_dict({f"metrics.{k}": v for k, v in state.items()})
        reference = self._track(_threaded_module(config))
        reference.load_state_dict(copy.deepcopy(state))
        for step in range(6, 9):
            isolated.update(_batch(step))
            reference.update(_batch(step))
        _assert_metrics_equal(
            self,
            isolated.async_compute().resolve(),
            reference.async_compute().resolve(),
        )
        self.assertEqual(
            sorted(parent.state_dict()),
            sorted(f"metrics.{k}" for k in reference.state_dict()),
        )

    def test_non_strict_load_reports_same_keys_as_threaded(self) -> None:
        source = self._track(
            _threaded_module(
                _metrics_config(
                    metrics=(
                        RecMetricEnum.NE,
                        RecMetricEnum.CTR,
                        RecMetricEnum.ACCURACY,
                    )
                )
            )
        )
        for step in range(6):
            source.update(_batch(step))
        source.sync()
        state = source.state_dict()
        config = _metrics_config()
        threaded = self._track(_threaded_module(config))
        isolated = self._track(_process_module(config))

        expected = threaded.load_state_dict(copy.deepcopy(state), strict=False)
        actual = isolated.load_state_dict(copy.deepcopy(state), strict=False)

        self.assertEqual(
            (sorted(actual.missing_keys), sorted(actual.unexpected_keys)),
            (sorted(expected.missing_keys), sorted(expected.unexpected_keys)),
        )
        for step in range(6, 9):
            threaded.update(_batch(step))
            isolated.update(_batch(step))
        _assert_metrics_equal(
            self,
            isolated.async_compute().resolve(),
            threaded.async_compute().resolve(),
        )

    def test_non_strict_recursive_load_reports_same_keys_as_threaded(self) -> None:
        source = self._track(
            _threaded_module(
                _metrics_config(
                    metrics=(
                        RecMetricEnum.NE,
                        RecMetricEnum.CTR,
                        RecMetricEnum.ACCURACY,
                    )
                )
            )
        )
        for step in range(6):
            source.update(_batch(step))
        source.sync()
        state = {f"metrics.{k}": v for k, v in source.state_dict().items()}
        config = _metrics_config()
        threaded_parent = torch.nn.Module()
        threaded_parent.metrics = self._track(_threaded_module(config))
        isolated_parent = torch.nn.Module()
        isolated_parent.metrics = self._track(_process_module(config))

        expected = threaded_parent.load_state_dict(copy.deepcopy(state), strict=False)
        actual = isolated_parent.load_state_dict(copy.deepcopy(state), strict=False)

        self.assertEqual(
            (sorted(actual.missing_keys), sorted(actual.unexpected_keys)),
            (sorted(expected.missing_keys), sorted(expected.unexpected_keys)),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_ring_pin_failure_raises(self) -> None:
        with patch.object(torch.cuda, "cudart") as cudart:
            cudart.return_value.cudaHostRegister.return_value = 1
            with self.assertRaisesRegex(RecMetricException, "cudaHostRegister"):
                _process_module(
                    _metrics_config(), model_out_device=torch.device("cuda")
                )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_outputs_packed_by_dtype_match_threaded(self) -> None:
        config = _metrics_config()
        threaded = self._track(_threaded_module(config))
        isolated = self._track(
            _process_module(config, model_out_device=torch.device("cuda"))
        )
        for step in range(12):
            batch = _batch(step)
            cuda_batch = {k: v.to("cuda") for k, v in batch.items()}
            cuda_batch["extra_f64"] = torch.full(
                (BATCH_SIZE,), float(step), device="cuda"
            ).double()
            cuda_batch["extra_i64"] = torch.arange(BATCH_SIZE, device="cuda") + step
            threaded.update(batch)
            isolated.update(cuda_batch)
        _assert_metrics_equal(
            self,
            isolated.async_compute().resolve(),
            threaded.async_compute().resolve(),
        )
        layouts = list(isolated._writer._layouts.values())
        self.assertEqual(
            len(layouts[0].groups), len({t.dtype for t in cuda_batch.values()})
        )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_outputs_overwritten_after_update_are_not_observed(self) -> None:
        config = _metrics_config()
        threaded = self._track(_threaded_module(config))
        isolated = self._track(
            _process_module(config, model_out_device=torch.device("cuda"))
        )
        for step in range(8):
            batch = _batch(step)
            cuda_batch = {k: v.to("cuda") for k, v in batch.items()}
            threaded.update(batch)
            isolated.update(cuda_batch)
            for tensor in cuda_batch.values():
                tensor.fill_(float("nan"))
        _assert_metrics_equal(
            self,
            isolated.async_compute().resolve(),
            threaded.async_compute().resolve(),
        )

    def test_idle_worker_sleeps_instead_of_polling(self) -> None:
        isolated = self._track(_process_module(_metrics_config()))
        isolated.update(_batch(0))
        isolated.async_compute().resolve()
        tick = os.sysconf("SC_CLK_TCK")

        def thread_cpu() -> dict[str, float]:
            out: dict[str, float] = {}
            for tid in os.listdir(f"/proc/{isolated._process.pid}/task"):
                with open(f"/proc/{isolated._process.pid}/task/{tid}/stat") as f:
                    stat = f.read()
                name = stat[stat.index("(") + 1 : stat.rindex(")")]
                fields = stat.rsplit(") ", 1)[1].split()
                out[f"{name}:{tid}"] = (int(fields[11]) + int(fields[12])) / tick
            return out

        threads_before = thread_cpu()
        time.sleep(3.0)
        threads_after = thread_cpu()
        # Library service threads started by imports in the spawned process are
        # out of scope; the worker's own threads must not poll.
        ours = ("metric_worker", "metric_update", "metric_compute")
        busy = {
            name: round(cpu - threads_before.get(name, 0.0), 2)
            for name, cpu in threads_after.items()
            if name.startswith(ours)
        }
        self.assertTrue(busy)
        self.assertLess(sum(busy.values()), 0.1, msg=f"worker threads: {busy}")

    def test_micro_batching_fails_closed(self) -> None:
        isolated = self._track(_process_module(_metrics_config()))
        isolated.set_under_micro_batching(False)
        with self.assertRaisesRegex(RecMetricException, "gradient accumulation"):
            isolated.set_under_micro_batching(True)
        self.assertFalse(isolated._process.is_alive())
        with self.assertRaisesRegex(RecMetricException, "gradient accumulation"):
            isolated.update_micro_batch(_batch(0))

    def test_concurrent_control_requests_keep_pipe_framing(self) -> None:
        isolated = self._track(_process_module(_metrics_config()))
        isolated.update(_batch(0))
        expected = set(isolated.state_dict())
        errors: list[BaseException] = []

        def request_states() -> None:
            try:
                for _ in range(20):
                    self.assertEqual(set(isolated.state_dict()), expected)
            except (AssertionError, RecMetricException) as error:
                errors.append(error)

        threads = [threading.Thread(target=request_states) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        self.assertEqual(errors, [])

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_slots_publish_without_ring_pressure(self) -> None:
        isolated = self._track(
            _process_module(_metrics_config(), model_out_device=torch.device("cuda"))
        )
        for step in range(3):
            isolated.update({k: v.to("cuda") for k, v in _batch(step).items()})
        deadline = time.monotonic() + 30
        while isolated._writer.consumed() < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(isolated._writer.published(), 3)
        self.assertEqual(isolated._writer.consumed(), 3)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_ring_publisher_failure_fails_the_trainer(self) -> None:
        isolated = self._track(
            _process_module(_metrics_config(), model_out_device=torch.device("cuda"))
        )
        with patch.object(
            torch.cuda.Event, "synchronize", side_effect=RuntimeError("cuda fault")
        ):
            with self.assertRaisesRegex(RecMetricException, "ring publisher failed"):
                for step in range(10_000):
                    isolated.update(
                        {k: v.to("cuda") for k, v in _batch(step % 4).items()}
                    )

    def test_publisher_failure_fails_pending_and_later_calls(self) -> None:
        isolated = self._track(_process_module(_metrics_config()))
        pending = _WorkerFuture(isolated)
        isolated._pending[-2] = pending
        isolated._writer._on_error(RecMetricException("ZORM ring publisher failed"))
        with self.assertRaisesRegex(RecMetricException, "ring publisher failed"):
            pending.result(timeout=5)
        with self.assertRaisesRegex(RecMetricException, "ring publisher failed"):
            isolated.update(_batch(0))


class MetricWorkerShutdownTest(unittest.TestCase):
    def _worker(
        self, module: MagicMock, drain: MagicMock, reply: MagicMock
    ) -> _MetricWorker:
        worker = _MetricWorker.__new__(_MetricWorker)
        worker._module = module
        worker._drain_updates = drain
        worker.reply = reply
        return worker

    def test_hung_module_shutdown_is_bounded(self) -> None:
        release = threading.Event()
        module, reply = MagicMock(), MagicMock()
        module.shutdown.side_effect = lambda: release.wait(60)
        worker = self._worker(module, MagicMock(), reply)
        with patch(
            "torchrec.metrics.cpu_offloaded_metric_worker._SHUTDOWN_BOUND_SEC", 0.2
        ), self.assertRaisesRegex(RuntimeError, "shutdown exceeded"):
            worker._on_shutdown(ControlMessage(Control.SHUTDOWN, 0, req_id=1))
        release.set()
        reply.assert_not_called()

    def test_drain_failure_still_shuts_down_module_and_raises(self) -> None:
        module, reply = MagicMock(), MagicMock()
        drain = MagicMock(side_effect=ValueError("bad slot"))
        worker = self._worker(module, drain, reply)
        with self.assertRaisesRegex(ValueError, "bad slot"):
            worker._on_shutdown(ControlMessage(Control.SHUTDOWN, 0, req_id=1))
        module.shutdown.assert_called_once()
        reply.assert_not_called()


class MetricWorkerControlTest(unittest.TestCase):
    def _worker(self, reply: MagicMock) -> _MetricWorker:
        worker = _MetricWorker.__new__(_MetricWorker)
        worker.reply = reply
        worker._handlers = {Control.RING: worker._on_ring}
        return worker

    def test_missing_handler_fails_at_startup(self) -> None:
        handlers = {kind: MagicMock() for kind in Control if kind is not Control.RESET}
        with self.assertRaisesRegex(NotImplementedError, "RESET"):
            require_handlers(handlers, Control)

    def test_failed_ring_switch_is_fatal(self) -> None:
        reply = MagicMock()
        worker = self._worker(reply)
        worker._set_ring = MagicMock(side_effect=ValueError("bad ring"))
        with self.assertRaisesRegex(ValueError, "bad ring"):
            worker._handle_control(
                ControlMessage(Control.RING, 5, payload=(MagicMock(),))
            )
        reply.assert_not_called()
