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
"""Diff two HF-format checkpoints (weight-space comparison between RL steps).

Streams tensor-by-tensor from safetensors (a 32B pair fits in a few GB of
RAM), reporting per-tensor and per-layer-group deltas:

    relative delta   ||b - a|| / ||a||     how much the tensor moved
    cosine sim       cos(a, b)             direction change (1.0 = none)
    max |delta|      largest single-weight change

Typical use: compare consecutive GRPO steps to see where the policy update
landed. Steps whose batches had all-zero advantage should show ~zero delta
(only AdamW's decoupled weight decay, ~lr*wd ≈ 1e-8 relative); steps
containing solves show real gradient imprint, concentrated by layer.

Run where torch is installed (e.g. the Ray head pod):

    python3 diff_checkpoints.py \
        checkpoints/<proj>/<exp>/global_step_3/actor/huggingface \
        checkpoints/<proj>/<exp>/global_step_4/actor/huggingface \
        --top 20
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from collections import defaultdict

import torch
from safetensors import safe_open


def key_to_file_map(ckpt_dir: str) -> dict[str, str]:
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))  # noqa: PTH118,PTH207
    if not files:
        raise SystemExit(f"no .safetensors under {ckpt_dir}")
    mapping = {}
    for f in files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():  # noqa: SIM118
                mapping[k] = f
    return mapping


def load_tensor(path: str, key: str) -> torch.Tensor:
    with safe_open(path, framework="pt") as sf:
        return sf.get_tensor(key).to(torch.float32)


def layer_group(name: str) -> str:
    """Collapse parameter names into comparable groups (layer idx stripped)."""
    m = re.search(r"layers\.(\d+)\.(.+?)\.weight", name)
    if m:
        return m.group(2)  # e.g. self_attn.q_proj, mlp.down_proj
    return name.replace(".weight", "").replace(".bias", "")


def main() -> None:  # noqa: PLR0914
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ckpt_a")
    parser.add_argument("ckpt_b")
    parser.add_argument("--top", type=int, default=20, help="show N most-changed tensors")
    args = parser.parse_args()

    map_a, map_b = key_to_file_map(args.ckpt_a), key_to_file_map(args.ckpt_b)
    only_a, only_b = set(map_a) - set(map_b), set(map_b) - set(map_a)
    if only_a or only_b:
        print(f"WARNING: key mismatch - only in A: {len(only_a)}, only in B: {len(only_b)}")

    rows = []
    groups: dict[str, list[float]] = defaultdict(list)
    total_sq_delta = total_sq_a = 0.0
    for key in sorted(set(map_a) & set(map_b)):
        a, b = load_tensor(map_a[key], key), load_tensor(map_b[key], key)
        if a.shape != b.shape:
            print(f"SHAPE MISMATCH {key}: {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        delta = b - a
        na, nd = a.norm().item(), delta.norm().item()
        rel = nd / na if na else float("inf")
        cos = torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0
        ).item()
        rows.append((rel, nd, cos, delta.abs().max().item(), key))
        groups[layer_group(key)].append(rel)
        total_sq_delta += nd * nd
        total_sq_a += na * na

    global_rel = math.sqrt(total_sq_delta) / math.sqrt(total_sq_a)
    print(f"\n=== global: ||delta||/||A|| = {global_rel:.3e} "
          f"over {len(rows)} tensors ===")

    print(f"\ntop {args.top} tensors by relative delta:")
    print(f"{'rel delta':>12}{'cos sim':>10}{'max|d|':>12}  tensor")
    for rel, _nd, cos, mx, key in sorted(rows, reverse=True)[: args.top]:
        print(f"{rel:>12.3e}{cos:>10.6f}{mx:>12.3e}  {key}")

    print("\nmean relative delta by parameter group:")
    for g, vals in sorted(groups.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {sum(vals) / len(vals):.3e}  {g}  (n={len(vals)})")


if __name__ == "__main__":
    main()
