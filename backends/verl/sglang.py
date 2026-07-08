import logging

logger = logging.getLogger(__name__)


class SglangEnginePatch:
    """Patch SGLangHttpServer to allow metrics extraction via HTTP scraping."""

    @classmethod
    def apply(cls) -> None:
        # the imports are within the function to allow users to run workloads with a single backend.
        try:
            from verl.workers.rollout.sglang_rollout.async_sglang_server import (  # type: ignore[import-not-found]
                SGLangHttpServer,
            )

            from py_inference_scheduler.datalayer.metrics.verl.sglang import (
                get_sglang_routing_stats,
            )
        except ImportError as e:
            logger.info(
                "Skipping SGLang patch (normal on head node if SGLang is not installed): %s",
                e,
            )
            return

        try:
            SGLangHttpServer.get_routing_stats = get_sglang_routing_stats

            # Patch launch_server to create metrics directory on worker
            original_launch = SGLangHttpServer.launch_server

            async def patched_launch(self, *args, **kwargs):
                import os
                metrics_dir = os.environ.get('PROMETHEUS_MULTIPROC_DIR', '/tmp/metrics')  # noqa: S108
                os.makedirs(metrics_dir, exist_ok=True)  # noqa: PTH103
                return await original_launch(self, *args, **kwargs)

            SGLangHttpServer.launch_server = patched_launch

        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to apply SGLang patch: %s", e)
