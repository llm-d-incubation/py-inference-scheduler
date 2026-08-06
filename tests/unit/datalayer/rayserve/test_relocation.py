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

import pytest

pytest.importorskip("vllm")

from py_inference_scheduler.datalayer.rayserve.relocation import (
    merge_completion_responses,
    relocation_enabled,
)


def test_relocation_enabled_requires_exact_one(monkeypatch):
    monkeypatch.delenv("ENABLE_MOONCAKE_RELOCATION", raising=False)
    assert not relocation_enabled()
    monkeypatch.setenv("ENABLE_MOONCAKE_RELOCATION", "true")
    assert not relocation_enabled()
    monkeypatch.setenv("ENABLE_MOONCAKE_RELOCATION", "1")
    assert relocation_enabled()


def _completion(rid, text, token_ids, prompt_tokens, finish_reason, prompt_ids=None):
    return {
        "id": rid,
        "created": 111 if rid == "push" else 222,
        "model": "m",
        "prompt_token_ids": prompt_ids,
        "choices": [
            {
                "index": 0,
                "text": text,
                "token_ids": token_ids,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(token_ids),
            "total_tokens": prompt_tokens + len(token_ids),
        },
    }


def test_merge_completion_responses():
    push = _completion("push", "Hello ", [1, 2], 10, "abort", prompt_ids=[7, 8])
    pull = _completion("pull", "world", [3, 4, 5], 12, "stop")

    merged = merge_completion_responses(push, pull)

    assert merged["id"] == "push"
    assert merged["created"] == 111
    choice = merged["choices"][0]
    assert choice["text"] == "Hello world"
    assert choice["token_ids"] == [1, 2, 3, 4, 5]
    assert choice["finish_reason"] == "stop"
    # Client-visible accounting: original prompt only, d1+d2 completions.
    assert merged["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert merged["prompt_token_ids"] == [7, 8]
