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
"""Backfill W&B history from verl console logs.

verl's wandb history can be lost when the SDK->service channel dies during
long idle gaps between steps (observed on Ray + wandb-core 0.22: run exists,
config/stats synced, zero history rows). The console logger always has the
full per-step metrics, so this parses `step:N - key:value - ...` lines from a
Ray job log and (re)logs them into the W&B run.

Usage (WANDB_API_KEY must be set, e.g. on the Ray head pod):
    python3 backfill_wandb.py --log_file job.log \
        --project swe-rl-scheduler --run_id tgpawp8d
"""

from __future__ import annotations

import argparse
import pathlib
import re

STEP_LINE_RE = re.compile(r"\bstep:(\d+) - (.+)$")


def parse_step_lines(text: str) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    for line in text.splitlines():
        m = STEP_LINE_RE.search(line)
        if not m:
            continue
        metrics: dict[str, float] = {}
        for pair in m.group(2).split(" - "):
            key, sep, value = pair.rpartition(":")
            if not sep:
                continue
            try:
                metrics[key.strip()] = float(value)
            except ValueError:
                continue  # non-numeric tail fragment
        if metrics:
            steps[int(m.group(1))] = metrics
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run_id", required=True, help="existing W&B run id to backfill into")
    parser.add_argument("--entity", default=None)
    args = parser.parse_args()

    with pathlib.Path(args.log_file).open(errors="replace", encoding="utf-8") as f:  # noqa: FURB101
        steps = parse_step_lines(f.read())
    if not steps:
        raise SystemExit("no `step:N - k:v` lines found in log file")

    import wandb

    run = wandb.init(
        project=args.project, entity=args.entity, id=args.run_id, resume="allow",
        settings=wandb.Settings(silent=True),
    )
    for step in sorted(steps):
        wandb.log(steps[step], step=step)
        print(f"backfilled step {step}: {len(steps[step])} metrics")
    run.finish()
    print(f"done -> {run.url}")


if __name__ == "__main__":
    main()
