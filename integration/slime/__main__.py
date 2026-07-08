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
import pathlib
from collections.abc import Callable

import uvicorn
import yaml
from fastapi import FastAPI

from integration.slime.server import create_app
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.config import SchedulerConfig

_DEFAULT_CONFIG = "integration/slime/examples/scheduler.yaml"


def run(
    create_app: Callable[[Scheduler], FastAPI],
    *,
    description: str,
    framework: str,
    default_config: str = _DEFAULT_CONFIG,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="0.0.0.0", help="bind address")  # noqa: S104
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument(
        "--config",
        default=default_config,
        help="path to scheduler.yaml (defaults to the bundled config; run from the repo root)",
    )
    parser.add_argument("--log-level", default="info", help="uvicorn/log level")
    parser.add_argument(
        "--proc-title",
        default="scheduler",
        help=f"process title; keeps the scheduler out of {framework}'s `pkill -9 python` cleanup",
    )
    args = parser.parse_args()

    # These frameworks' run scripts begin with `pkill -9 python` ("for rerun the task"), which
    # would kill this scheduler (a python process). Renaming the process so that cleanup misses it
    try:
        import setproctitle
        setproctitle.setproctitle(args.proc_title)
    except ImportError:
        logging.getLogger(__name__).warning(
            "setproctitle not installed; scheduler may be killed by %s's `pkill -9 python`",
            framework,
        )

    logging.basicConfig(level=args.log_level.upper())

    with pathlib.Path(args.config).open(encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = SchedulerConfig.from_dict(config_dict)
    logging.getLogger(__name__).info("Loaded scheduler config: %s", config)

    app = create_app(Scheduler.new_with_config(config))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


def main() -> None:
    run(create_app, description="sampling scheduler for slime", framework="slime")


if __name__ == "__main__":
    main()
