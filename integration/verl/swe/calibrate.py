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
"""Calibration sweep: screen R2E training instances in the real sandbox env.

The grading spec (``expected_output_json``) was generated in R2E's original
runc/Docker environment; we grade under gVisor. This sweep runs the PRE-FIX
graded suite for every instance in an actual sandbox and drops instances
that would inject reward noise into GRPO groups. An instance is kept only if:

1. **keys match** - the collected test set exactly equals the spec's keys
   (catches layout/collection/name-mapping drift);
2. **deterministic** - two consecutive runs produce identical status maps
   (catches flaky tests);
3. **not already solved** - the pre-fix status map does NOT equal the spec
   (otherwise doing nothing earns reward 1);
4. **within budget** - the suite finishes inside the grading time cap.

Emits one JSON line per instance plus a keep-list. Usage:

    uv run --with kubernetes --with pandas --with pyarrow python \
        -m integration.verl.swe.calibrate --parquet /tmp/swe_train.parquet \
        --out /tmp/calibration.jsonl --sample 8 --concurrency 4

Run from the repo root. Full sweep: drop --sample, raise --concurrency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from integration.verl.swe.grader import R2E_TEST_CMD, grade_r2e, parse_junitxml
from integration.verl.swe.sandbox import SandboxClient, SandboxError, get_thread_client

TIME_CAP_S = 300


def sandbox_name(instance_id: str) -> str:
    slug = re.sub(r"[^a-z0-9-]", "-", instance_id.lower()).strip("-")[:40]
    digest = hashlib.sha1(instance_id.encode()).hexdigest()[:6]  # noqa: S324
    return f"cal-{slug}-{digest}"


def run_suite(client: SandboxClient, name: str) -> tuple[dict[str, str], float]:
    start = time.monotonic()
    rc, out = client.exec(name, R2E_TEST_CMD, timeout=TIME_CAP_S + 30)
    elapsed = time.monotonic() - start
    if rc == 124 or elapsed > TIME_CAP_S:  # noqa: PLR2004
        raise TimeoutError(f"suite exceeded cap ({elapsed:.0f}s)")
    xml_start = out.find("<?xml")
    if xml_start < 0:
        raise ValueError(f"no junit xml in output (rc={rc}): {out[-400:]}")
    return parse_junitxml(out[xml_start:]), elapsed


def calibrate_instance(namespace: str, extra_info: dict) -> dict:
    client = get_thread_client(namespace)
    instance_id = extra_info["instance_id"]
    name = sandbox_name(instance_id)
    result: dict = {"instance_id": instance_id, "ok": False, "reason": ""}
    try:
        expected = json.loads(extra_info["expected_output_json"])
        client.create(name, extra_info["docker_image"])
        result["startup_s"] = round(client.wait_ready(name), 1)

        map1, elapsed1 = run_suite(client, name)
        map2, _ = run_suite(client, name)
        result["runtime_s"] = round(elapsed1, 1)
        result["n_tests"] = len(map1)

        grade = grade_r2e(map1, expected)
        result["keys_missing"] = len(grade.keys_missing)
        result["keys_extra"] = len(grade.keys_extra)
        result["prefix_mismatches"] = grade.mismatches[:20]

        if grade.keys_missing or grade.keys_extra:
            result["reason"] = "key-mismatch"
            result["missing_sample"] = grade.keys_missing[:5]
            result["extra_sample"] = grade.keys_extra[:5]
        elif map1 != map2:
            result["reason"] = "nondeterministic"
            result["flaky_tests"] = sorted(k for k in map1 if map1[k] != map2.get(k))[:10]
        elif grade.passed:
            result["reason"] = "already-solved"
        else:
            result["ok"] = True
            result["reason"] = "ok"
    except TimeoutError as e:
        result["reason"] = "timeout"
        result["error"] = str(e)
    except (SandboxError, ValueError) as e:
        result["reason"] = "sandbox-error"
        result["error"] = str(e)[:500]
    except Exception as e:  # keep the sweep alive; record and move on  # noqa: BLE001
        result["reason"] = "unexpected-error"
        result["error"] = f"{type(e).__name__}: {e}"[:500]
        result["traceback"] = traceback.format_exc()[-800:]
    finally:
        try:  # noqa: SIM105
            client.delete(name)
        except Exception:  # noqa: BLE001,S110
            pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True,
                        help="train parquet from prepare_swe_dataset.py")
    parser.add_argument("--out", required=True, help="results JSONL path")
    parser.add_argument("--keep_list", default=None,
                        help="write kept instance_ids here (default: <out>.keep)")
    parser.add_argument("--sample", type=int, default=None, help="only the first N instances")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--namespace", default="agents-system")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet, columns=["extra_info"])
    infos = [i for i in df["extra_info"] if i["dataset_kind"] == "r2e"]
    if args.sample:
        infos = infos[: args.sample]
    print(f"Calibrating {len(infos)} instances, concurrency {args.concurrency}")

    counts: dict[str, int] = {}
    done = 0
    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    with pathlib.Path(args.out).open("w", encoding="utf-8") as f, pool:
        futures = [pool.submit(calibrate_instance, args.namespace, info) for info in infos]
        for fut in as_completed(futures):
            r = fut.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            counts[r["reason"]] = counts.get(r["reason"], 0) + 1
            done += 1
            if done % 25 == 0 or done == len(infos):
                print(f"[{done}/{len(infos)}] {counts}")

    keep_path = args.keep_list or args.out + ".keep"
    with pathlib.Path(args.out).open() as f, pathlib.Path(keep_path).open("w") as k:  # noqa: FURB103,PLW1514
        kept = [r["instance_id"] for r in map(json.loads, f) if r["ok"]]
        k.write("\n".join(kept) + "\n")
    print(f"Kept {len(kept)}/{len(infos)} -> {keep_path}")
    print(f"Reasons: {counts}")


if __name__ == "__main__":
    main()
