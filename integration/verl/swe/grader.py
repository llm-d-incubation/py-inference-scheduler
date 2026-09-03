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
"""Pure grading logic for SWE RL trajectories.

Grading semantics (validated in-sandbox 2026-07-28, see docs/swe_bench_guide.md):

- R2E rows: the per-test status map from running ``/r2e_tests`` must EXACTLY
  equal the instance's ``expected_output_json``. Some tests are expected to
  remain FAILED after a correct fix - "all tests pass" is the wrong check.
- SWE-bench rows: after applying ``test_patch`` and the candidate patch, all
  ``fail_to_pass`` and all ``pass_to_pass`` tests must pass.

Everything here is pure (parses text, compares dicts) so it unit-tests without
a cluster. Sandbox execution lives in ``sandbox.py``.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # noqa: S405
from dataclasses import dataclass, field

PASSED = "PASSED"
FAILED = "FAILED"
SKIPPED = "SKIPPED"

# Runs the R2E graded suite and emits the junit XML on stdout. `;` (not `&&`)
# so the XML is emitted even when tests fail (pytest exits nonzero).
R2E_TEST_CMD = (
    "cd /testbed && .venv/bin/python -m pytest /r2e_tests "
    "--junitxml=/tmp/r2e_report.xml -q >/dev/null 2>&1; cat /tmp/r2e_report.xml"
)

_CLASS_SEGMENT = re.compile(r"_*[A-Z]")


def spec_key(classname: str, name: str) -> str:
    """Map junit (classname, name) to an R2E spec key.

    Observed spec keys look like ``TestUrlDispatcher.test_add_invalid_path``
    or ``_BaseTest.test_copy`` while junit reports
    ``classname="test_1.TestUrlDispatcher"``. Rule: drop leading module-path
    segments (lowercase-ish); class segments start with an uppercase letter,
    optionally underscore-prefixed. If nothing remains the test is
    module-level and the bare test name is the key. Parametrized suffixes
    (``[...]``) are part of ``name`` and kept as-is.
    """
    segments = [s for s in classname.split(".") if s] if classname else []
    class_segments = []
    for seg in segments:
        if class_segments or _CLASS_SEGMENT.match(seg):
            class_segments.append(seg)
    key = name if not class_segments else ".".join([*class_segments, name])
    # R2E spec generation splits pytest node ids on "::", which mangles "::"
    # inside parametrized values (e.g. IPv6 "cafe::17" -> "cafe.17").
    # Normalize the same way so keys compare equal.
    return key.replace("::", ".")


def parse_junitxml(xml_text: str) -> dict[str, str]:
    """Parse pytest junit XML into {spec_key: PASSED|FAILED|SKIPPED}.

    Test-level ``error`` (collection/setup crash) is treated as FAILED - for
    grading purposes a test that cannot run did not pass.
    """
    root = ET.fromstring(xml_text)  # noqa: S314
    statuses: dict[str, str] = {}
    for case in root.iter("testcase"):
        key = spec_key(case.get("classname", ""), case.get("name", ""))
        if case.find("failure") is not None or case.find("error") is not None:
            status = FAILED
        elif case.find("skipped") is not None:
            status = SKIPPED
        else:
            status = PASSED
        statuses[key] = status
    return statuses


@dataclass
class GradeResult:
    passed: bool
    keys_missing: list[str] = field(default_factory=list)  # in spec, not in run
    keys_extra: list[str] = field(default_factory=list)  # in run, not in spec
    mismatches: list[dict] = field(default_factory=list)  # same key, wrong status


def grade_r2e(status_map: dict[str, str], expected: dict[str, str]) -> GradeResult:
    """Exact status match against expected_output_json (R2E semantics).

    Skipped tests are dropped before comparison: R2E specs exclude them, and
    platform-conditional tests (e.g. Windows-only) skip in our environment.
    """
    status_map = {k: v for k, v in status_map.items() if v != SKIPPED}
    missing = sorted(set(expected) - set(status_map))
    extra = sorted(set(status_map) - set(expected))
    mismatches = [
        {"test": k, "expected": expected[k], "got": status_map[k]}
        for k in sorted(set(expected) & set(status_map))
        if status_map[k] != expected[k]
    ]
    return GradeResult(
        passed=not missing and not extra and not mismatches,
        keys_missing=missing,
        keys_extra=extra,
        mismatches=mismatches,
    )


def grade_swebench(f2p_map: dict[str, str], p2p_map: dict[str, str]) -> GradeResult:
    """All fail_to_pass and all pass_to_pass tests must pass."""
    mismatches = [
        {"test": k, "expected": PASSED, "got": v}
        for m in (f2p_map, p2p_map)
        for k, v in sorted(m.items())
        if v != PASSED
    ]
    return GradeResult(passed=not mismatches, mismatches=mismatches)
