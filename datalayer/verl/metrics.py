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

import asyncio
import aiohttp
import logging
from typing import Sequence, Optional
from scheduling.framework import Endpoint
from datalayer.verl.datastore import InflightStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def fetch_verl_metrics(server_handles):
    """Placeholder for synchronous RPC or Prometheus-based metric fetching for Verl workers."""
    # TODO: Implement the logic migrated from verl_hook.py once polling is abstracted
    return {}


async def verl_metrics_polling_loop(endpoints: Sequence[Endpoint], inflight_store: InflightStore):
    """Asynchronously scrapes metrics to update endpoint queue sizes (50ms interval)"""
    _worker_urls = None
    logger.info("DEBUG: verl_metrics_polling_loop started!")

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                if _worker_urls is None:
                    discovered_urls = []
                    for ep in endpoints:
                        actor = ep.attributes.get("replica_obj")
                        if not actor:
                             logger.warning(f"Endpoint {ep.name} missing replica_obj")
                             discovered_urls.append(None)
                             continue
                        try:
                            ip, port = await actor.get_server_address.remote()
                            discovered_urls.append(f"http://{ip}:{port}/metrics")
                        except Exception as e:
                            logger.error(f"Failed to fetch IP for {ep.name}: {e}")
                            discovered_urls.append(None)
                    _worker_urls = discovered_urls
                    logger.info(f"Natively mapped Worker Metrics IPs: {_worker_urls}")

                # Map exactly to _worker_urls index to pair with endpoints
                async def fetch_worker_metrics(session, idx, url):
                    if url is None:
                        return
                    logger.info(f"DEBUG: fetching metrics from {url}")
                    try:
                        async with session.get(url, timeout=2.0) as response:
                            text = await response.text()
                            stats = {
                                "num_waiting_reqs": 0,
                                "num_running_reqs": 0,
                                "kv": 0.0,
                                "queue_len": 0
                            }

                            for line in text.split('\n'):
                                try:
                                    if line.startswith("vllm:num_requests_waiting"):
                                        stats["num_waiting_reqs"] = int(float(line.split(" ")[-1]))
                                        # logger.debug(f"Waiting requests for {endpoints[idx].name}: {stats['num_waiting_reqs']}")
                                    elif line.startswith("vllm:num_requests_running"):
                                        stats["num_running_reqs"] = int(float(line.split(" ")[-1]))
                                        # logger.debug(f"Running requests for {endpoints[idx].name}: {stats['num_running_reqs']}")
                                    elif line.startswith("vllm:kv_cache_usage_perc"):
                                        stats["kv"] = float(line.split(" ")[-1])
                                except IndexError:
                                    continue

                            endpoint = endpoints[idx]
                            local_inflight = inflight_store.get(endpoint.name)
                            stats["num_running_reqs"] = max(stats["num_running_reqs"], local_inflight)

                            stats["queue_len"] = stats["num_waiting_reqs"] + stats["num_running_reqs"]
                            endpoint.attributes["routing_stats"] = stats

                    except asyncio.TimeoutError:
                        logger.debug(f"Timeout connecting to {url}")
                    except Exception as e:
                        logger.error(f"Failed to scrape {url}: {e}")

                tasks = [fetch_worker_metrics(session, i, url) for i, url in enumerate(_worker_urls)]
                await asyncio.gather(*tasks)

            except Exception as e:
                logger.error(f"Metrics poll error: {e}")

            # Poll at 50ms - same as gateway
            await asyncio.sleep(0.05)
