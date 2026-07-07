import logging
import re
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


async def get_sglang_routing_stats(server) -> dict:
    """Fetches routing stats from SGLang server via its Prometheus metrics endpoint."""
    stats: dict[str, Any] = {
        "num_waiting_reqs": 0,
        "num_running_reqs": 0,
        "kv": 0.0,
        "error": None,
    }
    try:
        # Get server address and port (this is called inside the actor, so server is self)
        host, port = server.get_server_address()
        url = f"http://{host}:{port}/metrics"

        async with aiohttp.ClientSession() as session:  # noqa: SIM117
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:  # noqa: PLR2004
                    text = await response.text()

                    waiting = re.search(r'^sglang:num_queue_reqs(?:\{.*?\})?\s+([\d.]+)', text, re.MULTILINE)  # noqa: E501
                    running = re.search(r'^sglang:num_running_reqs(?:\{.*?\})?\s+([\d.]+)', text, re.MULTILINE)  # noqa: E501
                    kv = re.search(r'^sglang:token_usage(?:\{.*?\})?\s+([\d.]+)', text, re.MULTILINE)  # noqa: E501

                    if waiting:
                        stats["num_waiting_reqs"] = int(float(waiting.group(1)))
                    if running:
                        stats["num_running_reqs"] = int(float(running.group(1)))
                    if kv:
                        stats["kv"] = float(kv.group(1))
                else:
                    stats["error"] = f"HTTP error {response.status}"
    except Exception as e:
        stats["error"] = str(e)
        logger.exception("Failed to get SGLang routing stats")

    return stats
