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

from __future__ import annotations

from types import SimpleNamespace

from py_inference_scheduler.framework.helpers import prefill_tokens


def test_prefill_tokens_counts_ids_and_estimates_text():
    assert prefill_tokens([5, 6, 7]) == 3
    assert prefill_tokens("x" * 40) == 10
    assert prefill_tokens([{"role": "user", "content": "x" * 40}]) == 10
    assert prefill_tokens([SimpleNamespace(role="user", content="x" * 40)]) == 10
    assert prefill_tokens(None) == 0
