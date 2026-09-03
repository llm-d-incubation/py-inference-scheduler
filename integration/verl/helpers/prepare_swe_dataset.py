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
"""Preprocess SWE RL datasets into verl parquet format.

Train split:  R2E-Gym (default R2E-Gym/R2E-Gym-Lite), graded via its
              per-instance ``expected_output_json`` test spec.
Eval split:   SWE-bench Verified, graded via FAIL_TO_PASS + PASS_TO_PASS.

Rows follow the verl RLHFDataset conventions (see verl's
examples/data_preprocess/gsm8k_tool_agent_loop.py): top-level ``agent_name``
selects the agent loop, ``prompt`` is a chat-message list, and everything the
agent loop needs at rollout time (sandbox image, test spec) is JSON-encoded
inside ``extra_info`` so parquet schemas stay flat and identical across splits.

Training instances whose repo also appears in SWE-bench Verified are dropped
to avoid eval contamination.

Usage:
    python integration/verl/helpers/prepare_swe_dataset.py \
        --local_save_dir ~/data/swe

    # quick smoke test (streams instead of downloading full datasets)
    python integration/verl/helpers/prepare_swe_dataset.py \
        --local_save_dir /tmp/swe_data --max_train 50 --max_eval 20
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib

import datasets

TRAIN_SOURCE_DEFAULT = "R2E-Gym/R2E-Gym-Lite"
EVAL_SOURCE_DEFAULT = "princeton-nlp/SWE-bench_Verified"
AGENT_NAME_DEFAULT = "swe_agent"

# The scaffold prompt is expected to evolve alongside the SWEAgentLoop
# implementation; the loop owns the tool schema, this owns the task framing.
SYSTEM_PROMPT = (
    "You are an expert software engineer. You are working in a sandboxed "
    "checkout of a real repository, and you are given a GitHub issue to "
    "resolve. Explore the repository, locate the root cause, and make the "
    "minimal source changes needed to fix the issue. Modify regular source "
    "code only - do not edit or delete existing tests. Verify your fix by "
    "running the relevant tests before submitting."
)

USER_PROMPT_TEMPLATE = (
    "The repository is checked out in the current working directory.\n\n"
    "Resolve the following issue:\n\n<issue>\n{problem_statement}\n</issue>"
)


def swebench_image(instance_id: str, registry: str) -> str:
    """Official SWE-bench per-instance image ref.

    Registry convention: ``__`` in instance ids becomes ``_1776_`` in image
    names (docker repo names reject consecutive underscores).
    """
    return f"{registry}/sweb.eval.x86_64.{instance_id.replace('__', '_1776_').lower()}:latest"


def load_split(source: str, split: str, max_rows: int | None) -> datasets.Dataset:
    """Load a split; stream only the first max_rows when capped (smoke tests)."""
    if max_rows:
        stream = datasets.load_dataset(source, split=split, streaming=True)
        return datasets.Dataset.from_list(list(itertools.islice(iter(stream), max_rows)))
    return datasets.load_dataset(source, split=split)


def rewrite_image(image: str, rewrites: list[str]) -> str:
    """Apply --registry_rewrite OLD=NEW prefix substitutions (for AR mirrors)."""
    for rule in rewrites:
        old, _, new = rule.partition("=")
        if image.startswith(old):
            return new + image[len(old):]
    return image


def make_row(  # noqa: PLR0913
    *,
    split: str,
    index: int,
    agent_name: str,
    data_source: str,
    problem_statement: str,
    extra_info: dict,
) -> dict:
    return {
        "data_source": data_source,
        "agent_name": agent_name,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(problem_statement=problem_statement),
            },
        ],
        "ability": "swe",
        # Reward comes from running tests in the sandbox, not from a comparator.
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"split": split, "index": index, **extra_info},
    }


def process_r2e(example: dict, idx: int, agent_name: str, source: str, rewrites: list[str]) -> dict:
    instance_id = f"{example['repo_name']}-{example['commit_hash'][:12]}"
    return make_row(
        split="train",
        index=idx,
        agent_name=agent_name,
        data_source=source,
        problem_statement=example["problem_statement"],
        extra_info={
            "instance_id": instance_id,
            "dataset_kind": "r2e",
            "docker_image": rewrite_image(example["docker_image"], rewrites),
            "repo": example["repo_name"],
            "base_commit": example["commit_hash"],
            # R2E grading spec: {test_name: expected status after a correct fix}.
            "expected_output_json": example["expected_output_json"],
            "fail_to_pass": "",
            "pass_to_pass": "",
            "version": "",
            "test_patch": "",
            # Difficulty proxies (size of the gold fix) for curriculum
            # filtering; leak-safe - extra_info never reaches the prompt.
            "difficulty": "",
            "num_non_test_files": int(example["num_non_test_files"]),
            "num_non_test_lines": int(example["num_non_test_lines"]),
        },
    )


def process_swebench(  # noqa: PLR0913,PLR0917
    example: dict, idx: int, agent_name: str, source: str, registry: str, rewrites: list[str]
) -> dict:
    instance_id = example["instance_id"]
    return make_row(
        split="test",
        index=idx,
        agent_name=agent_name,
        data_source=source,
        problem_statement=example["problem_statement"],
        extra_info={
            "instance_id": instance_id,
            "dataset_kind": "swebench",
            "docker_image": rewrite_image(swebench_image(instance_id, registry), rewrites),
            "repo": example["repo"],
            "base_commit": example["base_commit"],
            "expected_output_json": "",
            # Already JSON-encoded lists in the source dataset; kept as-is.
            "fail_to_pass": example["FAIL_TO_PASS"],
            "pass_to_pass": example["PASS_TO_PASS"],
            "version": example["version"],
            # Official SWE-bench grading applies test_patch (it contains the
            # fail-to-pass tests) before running the suite. Gold `patch` is
            # deliberately NOT carried - no answers in the dataset.
            "test_patch": example["test_patch"],
            # Human-annotated difficulty bucket (e.g. "15 min - 1 hour").
            "difficulty": example.get("difficulty", ""),
            "num_non_test_files": -1,
            "num_non_test_lines": -1,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_source", default=TRAIN_SOURCE_DEFAULT)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_source", default=EVAL_SOURCE_DEFAULT)
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--agent_name", default=AGENT_NAME_DEFAULT)
    parser.add_argument("--local_save_dir", default="~/data/swe")
    parser.add_argument("--max_train", type=int, default=None, help="Cap train rows (smoke tests).")
    parser.add_argument("--max_eval", type=int, default=None, help="Cap eval rows (smoke tests).")
    parser.add_argument(
        "--swebench_registry",
        default="docker.io/swebench",
        help="Registry hosting the official SWE-bench per-instance images.",
    )
    parser.add_argument(
        "--registry_rewrite",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite image ref prefixes, e.g. after mirroring to Artifact "
        "Registry: docker.io/namanjain12=us-docker.pkg.dev/my-proj/swe. Repeatable.",
    )
    args = parser.parse_args()

    eval_ds = load_split(args.eval_source, args.eval_split, args.max_eval)
    train_ds = load_split(args.train_source, args.train_split, args.max_train)

    # Contamination filter: drop training instances from repos that appear in
    # the eval set. SWE-bench repos are "org/name"; R2E-Gym repo_name is bare.
    eval_repos = {r.split("/")[-1].lower() for r in eval_ds["repo"]}
    before = len(train_ds)
    train_ds = train_ds.filter(lambda ex: ex["repo_name"].lower() not in eval_repos)
    print(f"Contamination filter: dropped {before - len(train_ds)} of {before} train rows "
          f"(eval repos: {sorted(eval_repos)})")

    train_ds = train_ds.map(
        lambda ex, i: process_r2e(ex, i, args.agent_name, args.train_source, args.registry_rewrite),
        with_indices=True,
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda ex, i: process_swebench(
            ex, i, args.agent_name, args.eval_source, args.swebench_registry, args.registry_rewrite
        ),
        with_indices=True,
        remove_columns=eval_ds.column_names,
    )

    save_dir = os.path.expanduser(args.local_save_dir)  # noqa: PTH111
    pathlib.Path(save_dir).mkdir(exist_ok=True, parents=True)
    train_path = os.path.join(save_dir, "train.parquet")  # noqa: PTH118
    test_path = os.path.join(save_dir, "test.parquet")  # noqa: PTH118
    train_ds.to_parquet(train_path)
    eval_ds.to_parquet(test_path)

    print(f"Wrote {len(train_ds)} train rows -> {train_path}")
    print(f"Wrote {len(eval_ds)} eval rows  -> {test_path}")
    sample = train_ds[0]
    print("Sample train row (truncated):")
    print(json.dumps(sample, default=str)[:1200])


if __name__ == "__main__":
    main()
