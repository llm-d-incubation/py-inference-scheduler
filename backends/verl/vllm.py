from __future__ import annotations

import logging

from py_inference_scheduler.datalayer.metrics.verl.vllm import get_vllm_routing_stats

logger = logging.getLogger(__name__)


class VllmEnginePatch:
    """Monkey-patching vLLM V1 (0.11.0+) to allow metrics extraction."""

    @classmethod
    def apply(cls) -> None:
        try:
            from verl.workers.rollout.vllm_rollout.vllm_async_server import (  # type: ignore[import-not-found]
                vLLMHttpServer,
            )
            from vllm.ray import ray_env  # type: ignore[import-not-found]
        except ImportError as e:
            logger.info("Skipping vLLM patch (normal on head node if vLLM is not installed): %s", e)
            return

        try:
            # Patch get_env_vars_to_copy to include PROMETHEUS_MULTIPROC_DIR
            original_get_env_vars = ray_env.get_env_vars_to_copy

            def patched_get_env_vars(destination="DPEngineCoreActor"):
                vars_list = original_get_env_vars(destination)
                if "PROMETHEUS_MULTIPROC_DIR" not in vars_list:
                    vars_list.append("PROMETHEUS_MULTIPROC_DIR")
                return vars_list

            ray_env.get_env_vars_to_copy = patched_get_env_vars
            vLLMHttpServer.get_routing_stats = get_vllm_routing_stats

            # Patch launch_server to create metrics directory on worker
            original_launch = vLLMHttpServer.launch_server

            async def patched_launch(self, *args, **kwargs):
                import os
                metrics_dir = os.environ.get('PROMETHEUS_MULTIPROC_DIR', '/tmp/metrics')  # noqa: S108
                os.makedirs(metrics_dir, exist_ok=True)  # noqa: PTH103
                os.environ['PROMETHEUS_MULTIPROC_DIR'] = metrics_dir
                return await original_launch(self, *args, **kwargs)

            vLLMHttpServer.launch_server = patched_launch

        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to apply vLLM patch: %s", e)
