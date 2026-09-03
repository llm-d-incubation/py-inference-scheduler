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

from integration.verl.swe.grader import (
    FAILED,
    PASSED,
    SKIPPED,
    grade_r2e,
    grade_swebench,
    parse_junitxml,
    spec_key,
)

JUNIT_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="5">
    <testcase classname="test_1.TestUrlDispatcher" name="test_add_invalid_path" time="0.01"/>
    <testcase classname="test_1.TestUrlDispatcher" name="test_register_route_checks" time="0.01">
      <failure message="assert failed">traceback</failure>
    </testcase>
    <testcase classname="test_1" name="test_module_level" time="0.01"/>
    <testcase classname="r2e_tests.test_2.TestNested" name="test_param[a-1]" time="0.01">
      <error message="setup boom">traceback</error>
    </testcase>
    <testcase classname="test_1.TestUrlDispatcher" name="test_skip_me" time="0.0">
      <skipped message="skip"/>
    </testcase>
  </testsuite>
</testsuites>
"""


def test_spec_key_strips_module_path():
    assert spec_key("test_1.TestUrlDispatcher", "test_x") == "TestUrlDispatcher.test_x"
    assert spec_key("r2e_tests.test_2.TestNested",
        "test_param[a-1]") == "TestNested.test_param[a-1]"
    assert spec_key("test_1", "test_module_level") == "test_module_level"
    assert spec_key("", "test_bare") == "test_bare"
    # underscore-prefixed classes are class segments, not module segments
    assert spec_key("test_1._BaseTest", "test_copy") == "_BaseTest.test_copy"
    # R2E specs mangle "::" inside parametrized values into "."
    assert (
        spec_key("test_1", 'test_node["[2001:db8:cafe::17]"-x]')
        == 'test_node["[2001:db8:cafe.17]"-x]'
    )


def test_grade_r2e_ignores_skipped():
    expected = {"T.test_a": PASSED}
    # platform-conditional tests skip in our env and are absent from specs
    result = grade_r2e({"T.test_a": PASSED, "T.test_win_only": SKIPPED}, expected)
    assert result.passed and not result.keys_extra  # noqa: PT018
    # but a spec-expected test that SKIPS in our env is a real key gap
    gap = grade_r2e({"T.test_a": SKIPPED}, expected)
    assert not gap.passed and gap.keys_missing == ["T.test_a"]  # noqa: PT018


def test_parse_junitxml_statuses():
    statuses = parse_junitxml(JUNIT_XML)
    assert statuses["TestUrlDispatcher.test_add_invalid_path"] == PASSED
    assert statuses["TestUrlDispatcher.test_register_route_checks"] == FAILED
    assert statuses["test_module_level"] == PASSED
    assert statuses["TestNested.test_param[a-1]"] == FAILED  # error counts as FAILED
    assert statuses["TestUrlDispatcher.test_skip_me"] == SKIPPED
    assert len(statuses) == 5


def test_grade_r2e_exact_match_including_expected_failures():
    expected = {"T.test_a": PASSED, "T.test_b": FAILED}
    assert grade_r2e({"T.test_a": PASSED, "T.test_b": FAILED}, expected).passed
    # all-green is NOT a pass when the spec expects a failure
    result = grade_r2e({"T.test_a": PASSED, "T.test_b": PASSED}, expected)
    assert not result.passed
    assert result.mismatches == [{"test": "T.test_b", "expected": FAILED, "got": PASSED}]


def test_grade_r2e_key_drift_fails():
    expected = {"T.test_a": PASSED}
    missing = grade_r2e({}, expected)
    assert not missing.passed and missing.keys_missing == ["T.test_a"]  # noqa: PT018
    extra = grade_r2e({"T.test_a": PASSED, "T.test_new": PASSED}, expected)
    assert not extra.passed and extra.keys_extra == ["T.test_new"]  # noqa: PT018


def test_grade_swebench():
    assert grade_swebench({"t1": PASSED}, {"t2": PASSED, "t3": PASSED}).passed
    result = grade_swebench({"t1": FAILED}, {"t2": PASSED})
    assert not result.passed
    assert result.mismatches[0]["test"] == "t1"
