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

from fastapi import FastAPI

from integration.slime.__main__ import run
from integration.tunix.server import create_app
from py_inference_scheduler import Scheduler

_DEFAULT_CONFIG = "integration/tunix/examples/scheduler.yaml"


def _app(scheduler: Scheduler, args: argparse.Namespace) -> FastAPI:
    del args  # decision service has no extra knobs
    return create_app(scheduler)


def main() -> None:
    run(
        _app,
        description="rollout decision scheduler for tunix",
        framework="tunix",
        default_config=_DEFAULT_CONFIG,
    )


if __name__ == "__main__":
    main()
