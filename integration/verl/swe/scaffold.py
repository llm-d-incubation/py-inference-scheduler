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
"""Pure helpers for the SWE agent scaffold (no verl / kubernetes imports).

Protocol (mini-swe-agent style): each assistant turn contains reasoning plus
exactly one fenced bash block. The block is executed in the sandbox and its
output returned as the next user message. A block containing only ``submit``
ends the episode and triggers grading.
"""

from __future__ import annotations

import re

SUBMIT_COMMAND = "submit"

BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)

# Paths the agent is not allowed to influence at grading time. Chunks of the
# agent's patch touching these are stripped before pristine-sandbox grading.
TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*$|_test\.py$|(^|/)conftest\.py$|(^|/)r2e_tests(/|$)"
)

NO_COMMAND_OBSERVATION = (
    "No bash code block found in your reply. Respond with your reasoning and "
    "exactly one fenced bash block, e.g.:\n```bash\nls\n```\n"
    f"When you are completely done, submit with:\n```bash\n{SUBMIT_COMMAND}\n```"
)


def parse_bash_command(text: str) -> str | None:
    """Extract the command from the LAST fenced bash block, or None."""
    blocks = BASH_BLOCK_RE.findall(text)
    if not blocks:
        return None
    return blocks[-1].strip()


def is_submit(command: str) -> bool:
    return command.strip().lower() == SUBMIT_COMMAND


def truncate_output(text: str, max_chars: int) -> str:
    """Middle-truncate command output: keep head and tail, mark the elision."""
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return f"{text[:keep]}\n...[{len(text) - 2 * keep} chars truncated]...\n{text[-keep:]}"


def format_observation(returncode: int, output: str, max_chars: int) -> str:
    body = truncate_output(output, max_chars) if output else "(no output)"
    return f"<returncode>{returncode}</returncode>\n<output>\n{body}\n</output>"


def _patch_paths(chunk: str) -> list[str]:
    """Both sides of a `diff --git a/... b/...` chunk header, repo-relative."""
    m = re.match(r"diff --git a/(\S+) b/(\S+)", chunk)
    return [m.group(1), m.group(2)] if m else []


def filter_patch(diff_text: str, path_re: re.Pattern = TEST_PATH_RE) -> str:
    """Drop per-file chunks whose path matches path_re (test files).

    The remainder is what gets applied and graded in the pristine sandbox -
    a policy that edits tests gains nothing from it.
    """
    if not diff_text.strip():
        return ""
    kept = []
    for chunk in re.split(r"(?m)^(?=diff --git )", diff_text):
        if not chunk.strip():
            continue
        paths = _patch_paths(chunk)
        if paths and any(path_re.search(p) for p in paths):
            continue
        kept.append(chunk)
    return "".join(kept)


def build_system_prompt(base_prompt: str) -> str:
    """Append the tool protocol to the dataset's task-framing system prompt."""
    return (
        f"{base_prompt}\n\n"
        "## Interaction protocol\n"
        "You are in a live shell session in the repository sandbox. In every "
        "reply, briefly reason about the next step, then give EXACTLY ONE "
        "fenced bash block with one command to run, e.g.:\n"
        "```bash\ngrep -rn \"pattern\" src/ | head -20\n```\n"
        "The command's output will be returned to you. Each command runs in a "
        "fresh shell from the repository root: cd and environment variables do "
        "NOT persist between turns. Edit files with shell commands (heredocs, "
        "sed). Long outputs are truncated - prefer targeted commands (head, "
        "grep). When your fix is complete and verified, end the session with:\n"
        f"```bash\n{SUBMIT_COMMAND}\n```"
    )
