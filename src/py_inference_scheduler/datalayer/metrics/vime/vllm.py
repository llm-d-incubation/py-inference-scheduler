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

import logging
from typing import Any

import aiohttp

from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.datalayer.metrics.prometheus import empty_vllm_stats, parse_vllm
from py_inference_scheduler.framework import Endpoint

logger = logging.getLogger(__name__)


async def scrape_vllm_metrics(
    worker_url: str, session: aiohttp.ClientSession, timeout: float = 5.0
) -> dict[str, Any]:
    url = f"{worker_url}/metrics"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status != 200:  # noqa: PLR2004
                stats = empty_vllm_stats()
                stats["error"] = f"HTTP error {response.status}"
                return stats
            return parse_vllm(await response.text())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to scrape vLLM metrics from %s: %s", url, e)
        stats = empty_vllm_stats()
        stats["error"] = str(e)
        return stats


async def fetch_worker_metrics(
    ep: Endpoint, inflight_store: InflightStore, session: aiohttp.ClientSession
) -> None:
    url = ep.attributes.get("url")
    if not url:
        return
    stats = await scrape_vllm_metrics(str(url), session)
    ep.attributes["queue_len"] = inflight_store.get(ep.name)
    ep.attributes["routing_stats"] = stats
