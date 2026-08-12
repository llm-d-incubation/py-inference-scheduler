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
import os
import pathlib
from collections.abc import Callable

import yaml

from integration.vllm_router.adapter import VllmRouterSchedulerAdapter
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.config import SchedulerConfig

logger = logging.getLogger(__name__)

ENV_CONFIG = "ROUTER_CONFIG_PATH"
ENV_METRICS_INTERVAL_MS = "RLS_METRICS_INTERVAL_MS"

# Keeps running adapters reachable (poller stop, staleness introspection)
_live_adapters: list[VllmRouterSchedulerAdapter] = []


def make_policy(router_args: object = None) -> Callable:
    """Target of vllm-router's --external-policy-factory flag.

    Configuration arrives via env rather than router_args so the fork needs
    no scheduler-specific flags: ROUTER_CONFIG_PATH (scheduler.yaml, required)
    and RLS_METRICS_INTERVAL_MS (poller interval, default 100).
    Raises on missing/invalid config: launch must fail closed, the router's
    fallback policy is reserved for per-request failures.
    """
    config_path = os.environ.get(ENV_CONFIG)
    if not config_path:
        raise ValueError(f"{ENV_CONFIG} must point to a scheduler yaml config")
    with pathlib.Path(config_path).open(encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    if not isinstance(config_dict, dict):
        raise TypeError("Parsed configuration is not a valid dictionary.")
    config = SchedulerConfig.from_dict(config_dict)
    logger.info("Loaded scheduler config: %s", config)

    interval_ms = int(os.environ.get(ENV_METRICS_INTERVAL_MS, "100"))
    adapter = VllmRouterSchedulerAdapter(
        Scheduler.new_with_config(config), metrics_interval_ms=interval_ms
    )
    static_urls = getattr(router_args, "worker_urls", None)
    if static_urls:
        adapter.seed_workers([str(u) for u in static_urls])
    adapter.start()

    _live_adapters.append(adapter)
    return adapter.select
