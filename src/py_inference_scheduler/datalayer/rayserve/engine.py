# Copyright 2026 llm-d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from ray.llm._internal.serve.core.server.llm_server import LLMServer  # noqa: PLC2701
from ray.llm._internal.serve.engines.vllm.vllm_engine import VLLMEngine  # noqa: PLC2701
from vllm.v1.metrics.loggers import StatLoggerBase  # type: ignore[import-not-found]


class DirectKVCacheLogger(StatLoggerBase):
    def __init__(self, vllm_config: object, engine_idx: int = 0) -> None:
        # The factory signature mandates these arguments.
        self.target_dict: dict[str, float | int] | None = None

    def log(self) -> None:
        pass

    def log_engine_initialized(self) -> None:
        pass

    def record(
        self,
        scheduler_stats: object,
        iteration_stats: object,
        engine_idx: int = 0,
    ) -> None:
        if self.target_dict is not None and scheduler_stats is not None:
            self.target_dict["kv"] = getattr(scheduler_stats, "kv_cache_usage", 0.0)
            self.target_dict["num_waiting_reqs"] = getattr(
                scheduler_stats, "num_waiting_reqs", 0
            )
            self.target_dict["num_running_reqs"] = getattr(
                scheduler_stats, "num_running_reqs", 0
            )

        if self.target_dict is not None and iteration_stats is not None:
            self.target_dict["num_preempted"] += getattr(iteration_stats, "num_preempted_reqs", 0)


class MetricsAwareVLLMEngine(VLLMEngine):
    def _start_async_llm_engine(
        self,
        engine_args: object,
        engine_config: object,
        pg: object,
    ) -> object:
        self.live_metrics = {
            "kv": 0.0,
            "num_waiting_reqs": 0,
            "num_running_reqs": 0,
            "num_preempted": 0,
        }

        # vLLM expects a StatLoggerFactory, so use a closure that attaches
        # our metrics dict to each logger instance.
        def logger_factory(
            vllm_config: object, engine_idx: int = 0
        ) -> DirectKVCacheLogger:
            logger = DirectKVCacheLogger(vllm_config, engine_idx)
            logger.target_dict = self.live_metrics  # type: ignore[assignment]
            return logger

        # Needed for injecting the logger
        from vllm.v1.engine.async_llm import AsyncLLM  # type: ignore[import-not-found]
        from vllm.v1.executor.abstract import Executor  # type: ignore[import-not-found]

        engine_config.parallel_config.placement_group = pg  # type: ignore[attr-defined]
        executor_class = Executor.get_class(engine_config)

        # Return the engine client, passing the FACTORY instead of the instance
        return AsyncLLM(
            vllm_config=engine_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,  # type: ignore[attr-defined]
            stat_loggers=[logger_factory],
        )

    def record_routing_stats(self) -> dict[str, object]:
        # Ray natively expects this to return a dictionary
        res: dict[str, object] = {
            "kv": self.live_metrics.get("kv", 0.0),
            "num_waiting_reqs": self.live_metrics.get("num_waiting_reqs", 0),
            "num_running_reqs": self.live_metrics.get("num_running_reqs", 0),
            "num_preempted": self.live_metrics.get("num_preempted", 0),
        }

        if not hasattr(self, "_total_kv_tokens"):
            try:
                engine = self._engine_client
                vllm_config = getattr(engine, "vllm_config", None)
                if vllm_config:
                    num_blocks = vllm_config.cache_config.num_gpu_blocks
                    block_size = vllm_config.cache_config.block_size
                    self._total_kv_tokens = num_blocks * block_size
                else:
                    self._total_kv_tokens = -1
            except Exception as e:  # noqa: BLE001
                print(f"[METRICS ERROR] Failed to extract KV Cache size: {e}")
                self._total_kv_tokens = -1

        res["kv_cache_size"] = self._total_kv_tokens
        return res


class MetricsAwareLLMServer(LLMServer):
    _default_engine_cls = MetricsAwareVLLMEngine

    async def record_routing_stats(self) -> dict[str, object]:
        return self.engine.record_routing_stats()  # type: ignore[no-any-return]
