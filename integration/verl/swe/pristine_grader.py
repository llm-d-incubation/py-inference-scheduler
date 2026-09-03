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
"""Pristine-state grading: score an agent patch in a sandbox it never touched.

The rollout sandbox is agent-controlled (root), so grading there is
exploitable - a policy can delete failing tests or patch pytest. Instead the
agent's diff is filtered to non-test paths, applied in a FRESH sandbox from
the same image, and graded there:

- r2e rows:      run /r2e_tests, exact-match vs expected_output_json
- swebench rows: apply test_patch, then all FAIL_TO_PASS + PASS_TO_PASS
                 tests must pass (official grade order)

Synchronous (kubernetes client) - call from an executor thread.
"""

from __future__ import annotations

import json
import logging

from integration.verl.swe.grader import R2E_TEST_CMD, grade_r2e, parse_junitxml
from integration.verl.swe.sandbox import get_thread_client, make_name
from integration.verl.swe.scaffold import filter_patch

logger = logging.getLogger(__name__)

SWEBENCH_PY = "/opt/miniconda3/envs/testbed/bin/python"
SUITE_TIMEOUT_S = 330  # in-sandbox test cap (~5 min) plus slack


def _run_pytest_ids(client, name: str, ids_json: str, label: str) -> bool:
    """Run a JSON list of pytest ids in the sandbox; True iff all pass.

    Ids go via a newline file + xargs -0: the lists can be long and exec
    commands travel in the request URL.
    """
    ids = json.loads(ids_json)
    if not ids:
        return True
    client.write_file(name, f"/tmp/{label}_ids.txt", "\n".join(ids))  # noqa: S108
    rc, out = client.exec(
        name,
        f"cd /testbed && tr '\\n' '\\0' < /tmp/{label}_ids.txt | "
        f"xargs -0 {SWEBENCH_PY} -m pytest -q > /tmp/{label}.log 2>&1; echo RC=$?",
        timeout=SUITE_TIMEOUT_S,
    )
    return rc == 0 and "RC=0" in out


def grade_patch_pristine(  # noqa: PLR0911
    extra_info: dict, agent_diff: str, namespace: str = "agents-system"
) -> tuple[float, str]:
    """Returns (reward, reason). Never raises - grading failures are reward 0."""
    filtered = filter_patch(agent_diff)
    if not filtered.strip():
        return 0.0, "empty-patch"

    client = get_thread_client(namespace)
    name = make_name("grade", extra_info["instance_id"])
    try:
        client.create(name, extra_info["docker_image"])
        client.wait_ready(name)
        client.write_file(name, "/tmp/agent.patch", filtered)  # noqa: S108
        rc, out = client.exec(name, "cd /testbed && git apply --whitespace=nowarn /tmp/agent.patch")
        if rc != 0:
            return 0.0, f"patch-apply-failed: {out[-200:]}"

        if extra_info["dataset_kind"] == "r2e":
            rc, out = client.exec(name, R2E_TEST_CMD, timeout=SUITE_TIMEOUT_S)
            xml_start = out.find("<?xml")
            if xml_start < 0:
                return 0.0, f"no-junit-output (rc={rc})"
            expected = json.loads(extra_info["expected_output_json"])
            result = grade_r2e(parse_junitxml(out[xml_start:]), expected)
            if result.passed:
                return 1.0, "pass"
            return 0.0, (
                f"tests-mismatch ({len(result.mismatches)} wrong, "
                f"{len(result.keys_missing)} missing, {len(result.keys_extra)} extra)"
            )

        # swebench: official order - test_patch on top of the candidate patch
        client.write_file(name, "/tmp/test.patch", extra_info["test_patch"])  # noqa: S108
        rc, out = client.exec(name, "cd /testbed && git apply --whitespace=nowarn /tmp/test.patch")
        if rc != 0:
            return 0.0, f"test-patch-apply-failed: {out[-200:]}"
        if not _run_pytest_ids(client, name, extra_info["fail_to_pass"], "f2p"):
            return 0.0, "f2p-failed"
        if not _run_pytest_ids(client, name, extra_info["pass_to_pass"], "p2p"):
            return 0.0, "p2p-failed"
        return 1.0, "pass"  # noqa: TRY300
    except Exception as e:  # noqa: BLE001 - grading must not kill the rollout
        logger.warning("pristine grading failed for %s: %s", extra_info.get("instance_id"), e)
        return 0.0, f"grading-error: {type(e).__name__}: {e}"[:300]
    finally:
        try:  # noqa: SIM105
            client.delete(name)
        except Exception:  # noqa: BLE001,S110
            pass
