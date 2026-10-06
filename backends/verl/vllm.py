from __future__ import annotations

import logging

from py_inference_scheduler.datalayer.metrics.verl.vllm import get_vllm_routing_stats

logger = logging.getLogger(__name__)


class VllmEnginePatch:
    """Expose vLLM routing stats on verl's server actor over Ray RPC."""

    @classmethod
    def apply(cls) -> None:
        try:
            from verl.workers.rollout.vllm_rollout.vllm_async_server import (  # type: ignore[import-not-found]
                vLLMHttpServer,
            )
        except Exception as e:  # noqa: BLE001 - vLLM internals raise more than
            # ImportError on CPU-only nodes (e.g. triton AttributeError on the
            # Ray head); any import failure means "no vLLM here", so skip.
            logger.info(
                "Skipping vLLM patch (normal on head node if vLLM is not importable): %s", e
            )
            return

        # No PROMETHEUS_MULTIPROC_DIR: shared by a node's engines, it aggregates every /metrics.
        vLLMHttpServer.get_routing_stats = get_vllm_routing_stats
