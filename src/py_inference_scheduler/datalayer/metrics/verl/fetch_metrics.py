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

import logging

from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.framework import Endpoint

logger = logging.getLogger(__name__)


async def fetch_worker_metrics(ep: Endpoint, inflight_store: InflightStore) -> None:
    """Request stats from a single worker via RPC and update the endpoint attributes."""
    actor = ep.attributes.get("replica_obj")
    if not actor:
        return
    try:
        stats = await actor.get_routing_stats.remote()  # type: ignore[attr-defined]
        if stats.get("error"):
            logger.error("RPC stats error for %s: %s", ep.name, stats["error"])
        local_inflight = inflight_store.get(ep.name)
        ep.attributes["queue_len"] = local_inflight
        ep.attributes["routing_stats"] = {
            "num_waiting_reqs": stats.get("num_waiting_reqs", 0),
            "num_running_reqs": stats.get("num_running_reqs", 0),
            "kv": stats.get("kv", 0.0),
            "error": stats.get("error", None),
        }
    except Exception:
        logger.exception("Failed to scrape RPC metrics for %s", ep.name)
