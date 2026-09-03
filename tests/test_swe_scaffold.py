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

from integration.verl.swe.scaffold import (
    filter_patch,
    format_observation,
    is_submit,
    parse_bash_command,
    truncate_output,
)


def test_parse_bash_command_variants():
    assert parse_bash_command("thinking...\n```bash\nls -la\n```") == "ls -la"
    assert parse_bash_command("```sh\ngrep -rn foo .\n```\n") == "grep -rn foo ."
    assert parse_bash_command("```\necho bare fence\n```") == "echo bare fence"
    # last block wins when several are present
    two = "```bash\nfirst\n```\ntext\n```bash\nsecond\n```"
    assert parse_bash_command(two) == "second"
    assert parse_bash_command("no code block here") is None


def test_is_submit():
    assert is_submit("submit")
    assert is_submit("  SUBMIT \n")
    assert not is_submit("git submit")


def test_truncate_output_middle():
    text = "A" * 600 + "B" * 600
    out = truncate_output(text, 200)
    assert len(out) < 300
    assert out.startswith("A" * 100)
    assert out.endswith("B" * 100)
    assert "truncated" in out
    assert truncate_output("short", 200) == "short"


def test_format_observation():
    obs = format_observation(1, "boom", 100)
    assert "<returncode>1</returncode>" in obs and "boom" in obs  # noqa: PT018
    assert "(no output)" in format_observation(0, "", 100)


def _chunk(path: str, body: str = "-old\n+new\n") -> str:
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n{body}"


def test_filter_patch_strips_test_paths():
    diff = (
        _chunk("aiohttp/web_urldispatcher.py")
        + _chunk("tests/test_urldispatch.py")
        + _chunk("r2e_tests/test_1.py")
        + _chunk("src/conftest.py")
        + _chunk("src/util_test.py")
        + _chunk("src/testing/helpers.py")
    )
    kept = filter_patch(diff)
    assert "web_urldispatcher" in kept
    for stripped in ("tests/test_urldispatch", "r2e_tests", "conftest", "util_test",
        "testing/helpers"):
        assert stripped not in kept


def test_filter_patch_keeps_lookalikes_and_handles_empty():
    # "contest.py" and "latest.py" are not test paths
    diff = _chunk("src/contest.py") + _chunk("src/latest.py")
    assert filter_patch(diff) == diff
    assert filter_patch("") == ""
    assert filter_patch("   \n") == ""
