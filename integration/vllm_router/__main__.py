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

import argparse
import logging
import os

from integration.vllm_router.factory import ENV_CONFIG, ENV_METRICS_INTERVAL_MS

FACTORY_SPEC = "integration.vllm_router.factory:make_policy"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="vllm-router driven by the rl-scheduler (external policy)",
        epilog="All other arguments are forwarded to vllm-router; see `vllm-router --help`.",
    )
    parser.add_argument(
        "--external-scheduler-config",
        required=True,
        help="path to the scheduler yaml config",
    )
    parser.add_argument(
        "--external-metrics-interval-ms",
        type=int,
        default=100,
        help="worker /metrics polling interval for the MetricsPoller",
    )
    parser.add_argument("--log-level", default="info", help="python-side log level")
    args, router_argv = parser.parse_known_args(argv)

    # timestamps matter here: adapter/poller lines interleave with the Rust
    # router's own timestamped log and phase correlation depends on them
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if os.environ.get("RLS_DECISION_LOG") == "1":
        # per-request candidate stats without global DEBUG noise; costs one
        # formatted log line per request while enabled - not for benchmarking
        from py_inference_scheduler.core.scheduler import decision_logger

        decision_logger.setLevel(logging.DEBUG)
    os.environ[ENV_CONFIG] = args.external_scheduler_config
    os.environ[ENV_METRICS_INTERVAL_MS] = str(args.external_metrics_interval_ms)

    try:
        from vllm_router.launch_router import launch_router, parse_router_args
    except ImportError as e:
        raise SystemExit(
            "vllm-router (the Rust router fork) is not installed in this environment; "
            "install its wheel into the venv first"
        ) from e

    router_args = parse_router_args(router_argv)
    router_args.external_policy_factory = FACTORY_SPEC
    if router_args.external_fallback_policy is None:
        router_args.external_fallback_policy = "round_robin"
    launch_router(router_args)


if __name__ == "__main__":
    main()
