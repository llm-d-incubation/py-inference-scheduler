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

from ray.llm._internal.serve.engines.vllm.vllm_engine import VLLMEngine
from ray.llm._internal.serve.core.server.llm_server import LLMServer
from vllm.v1.metrics.loggers import StatLoggerBase


class DirectKVCacheLogger(StatLoggerBase):
    def __init__(self, vllm_config, engine_idx=0):
        # The factory signature mandates these arguments.
        self.target_dict = None

    def log(self):
        pass

    def log_engine_initialized(self):
        pass

    def record(self, scheduler_stats, iteration_stats, engine_idx=0):
        if self.target_dict is not None and scheduler_stats is not None:
            self.target_dict["kv"] = getattr(scheduler_stats, "kv_cache_usage", 0.0)
            self.target_dict["num_waiting_reqs"] = getattr(scheduler_stats, "num_waiting_reqs", 0)
            self.target_dict["num_running_reqs"] = getattr(scheduler_stats, "num_running_reqs", 0)


class MetricsAwareVLLMEngine(VLLMEngine):
    def _start_async_llm_engine(self, engine_args, engine_config, pg):
        self.live_metrics = {
            "kv": 0.0,
            "num_waiting_reqs": 0,
            "num_running_reqs": 0
        }

        # vLLM expects a StatLoggerFactory (a Callable that returns the Logger) - we create a closure that instantiates the logger and attaches our dict
        def logger_factory(vllm_config, engine_idx=0):
            logger = DirectKVCacheLogger(vllm_config, engine_idx)
            logger.target_dict = self.live_metrics
            return logger

        # Needed for injecting the logger
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.executor.abstract import Executor

        engine_config.parallel_config.placement_group = pg
        executor_class = Executor.get_class(engine_config)

        # Return the engine client, passing the FACTORY instead of the instance
        return AsyncLLM(
            vllm_config=engine_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            stat_loggers=[logger_factory]
        )

    def record_routing_stats(self):
        # Ray natively expects this to return a dictionary
        return {
            "kv": self.live_metrics.get("kv", 0.0),
            "num_waiting_reqs": self.live_metrics.get("num_waiting_reqs", 0),
            "num_running_reqs": self.live_metrics.get("num_running_reqs", 0)
        }


class MetricsAwareLLMServer(LLMServer):
    _default_engine_cls = MetricsAwareVLLMEngine

    async def record_routing_stats(self):
        return self.engine.record_routing_stats()
