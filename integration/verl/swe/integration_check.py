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
"""End-to-end SWEAgentLoop check without GPUs: scripted model, real everything else.

Runs ON the Ray head pod (needs verl, transformers, kubernetes, and in-cluster
RBAC). Drives SWEAgentLoop.run() with a scripted server_manager that (1)
explores, (2) writes the instance's gold fix via heredoc, (3) submits — then
asserts the pristine-grading path returns reward 1.0. Exercises: verl base
class against the installed build, tokenization discipline, sandbox lifecycle,
observation flow, baseline commit, diff extraction/filtering, fresh-sandbox
grading.

Usage (see docs/swe_bench_guide.md):
    PYTHONPATH=/tmp/swe_itest python3 -m integration.verl.swe.integration_check \
        --parquet /home/ray/data/swe/train.parquet --instance aiohttp-f0d74880deec
"""

from __future__ import annotations

import argparse
import asyncio
import json
import types
import urllib.request

R2E_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=R2E-Gym%2FR2E-Gym-Lite"
    "&config=default&split=train&offset={offset}&length=1"
)


class ScriptedServerManager:
    """Feeds pre-scripted assistant turns; records what the loop sent."""

    def __init__(self, tokenizer, turns: list[str]):  # noqa: ANN204
        self.tokenizer = tokenizer
        self.turns = list(turns)
        self.calls: list[int] = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):  # noqa: ANN201
        self.calls.append(len(prompt_ids))
        text = self.turns.pop(0)
        return types.SimpleNamespace(
            token_ids=self.tokenizer.encode(text, add_special_tokens=False),
            log_probs=None,
        )


def fetch_gold_fix(offset: int) -> dict[str, str]:
    """{repo-relative path: new content} for the instance's gold commit (non-test files)."""
    with urllib.request.urlopen(R2E_ROWS_URL.format(offset=offset), timeout=60) as r:  # noqa: S310
        row = json.load(r)["rows"][0]["row"]
    files = {}
    for d in json.loads(row["parsed_commit_content"])["file_diffs"]:
        plus = d["plus_file"]
        path = (plus if isinstance(plus, str) else plus.get("path", "")).removeprefix("b/")
        files[path] = d["new_file_content"]
    return row["repo_name"], row["commit_hash"], files


async def main() -> int:  # noqa: PLR0914
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--instance", default="aiohttp-f0d74880deec")
    parser.add_argument("--row_offset", type=int, default=0,
        help="row of the instance in the HF dataset")
    args = parser.parse_args()

    import pandas as pd
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.chat_template import initialize_system_prompt

    from integration.verl.swe.swe_agent_loop import SWEAgentLoop

    df = pd.read_parquet(args.parquet)
    row = next(r for _, r in df.iterrows() if r["extra_info"]["instance_id"] == args.instance)
    extra_info = dict(row["extra_info"])
    raw_prompt = [dict(m) for m in row["prompt"]]

    repo, commit, gold_files = fetch_gold_fix(args.row_offset)
    base = extra_info["base_commit"]
    assert commit.startswith(base[:12]) or base.startswith(commit[:12]), (  # noqa: S101
        f"row_offset points at {repo}@{commit[:12]}, expected {args.instance}"
    )
    src_files = {p: c for p, c in gold_files.items() if not p.startswith("tests/")}
    print(f"gold fix: {len(gold_files)} files, {len(src_files)} non-test")

    write_cmds = "\n".join(
        f"cat > {path} << 'SWE_INTEGRATION_EOF'\n{content}SWE_INTEGRATION_EOF"
        for path, content in src_files.items()
    )
    turns = [
        "Let me look at the repository layout first.\n```bash\nls | head -5\n```",
        f"I found the issue; applying the fix now.\n```bash\n{write_cmds}\n```",
        "The fix is complete.\n```bash\nsubmit\n```",
    ]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    agent = SWEAgentLoop.__new__(SWEAgentLoop)
    agent.tokenizer = tokenizer
    agent.processor = None
    agent.apply_chat_template_kwargs = {}
    agent.system_prompt = initialize_system_prompt(tokenizer)
    agent.rollout_config = OmegaConf.create({"prompt_length": 8192, "response_length": 16384})
    agent.response_length = 16384
    agent.max_assistant_turns = 8
    agent._init_tunables()
    agent.loop = asyncio.get_running_loop()
    agent.server_manager = ScriptedServerManager(tokenizer, turns)

    output = await agent.run({"temperature": 1.0}, raw_prompt=raw_prompt, extra_info=extra_info)

    n_obs = sum(1 for m in output.response_mask if m == 0)
    n_gen = sum(output.response_mask)
    print(
        f"reward={output.reward_score} extra={output.extra_fields} num_turns={output.num_turns} "
        f"prompt={len(output.prompt_ids)} response={len(output.response_ids)} "
        f"gen_tokens={n_gen} obs_tokens={n_obs}"
    )
    ok = (
        output.reward_score == 1.0  # noqa: RUF069
        and output.extra_fields.get("swe_submitted") is True
        and len(output.response_ids) == len(output.response_mask)
        and n_obs > 0
    )
    print("INTEGRATION CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
