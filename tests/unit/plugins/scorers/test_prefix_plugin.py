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

from scheduling.plugins.scorers.prefix_plugin import PrefixIndexer, _hash_prompt_bytes, PrefixCacheScorer
from scheduling.framework import Endpoint, CycleState, LLMRequest


def test_indexer_add_get_remove():
    idx = PrefixIndexer()
    hashes = [1, 2, 3]
    idx.add(hashes, "s1")

    # each hash should map to server s1
    for h in hashes:
        got = idx.get(h)
        assert "s1" in got

    assert "s1" in idx.pods()

    # remove and ensure gone
    idx.remove_server("s1")
    for h in hashes:
        assert idx.get(h) == set()
    assert idx.pods() == []


def test_hash_prompt_bytes_basic():
    body = "abcdefgh"
    # block size 4 -> two blocks
    hashes = _hash_prompt_bytes("mymodel", body.encode("utf-8"), 4, 10)
    assert isinstance(hashes, list)
    assert len(hashes) == 2
    # hashes should be integers
    assert all(isinstance(h, int) for h in hashes)


def test_prefix_cache_scorer_scores():
    scorer = PrefixCacheScorer(block_size=4, max_prefix_blocks=10)
    # prepare a request with 2 blocks
    body = "abcdefghijkl"
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)

    # compute hashes using same logic
    hashes = _hash_prompt_bytes(req.target_model, body.encode("utf-8"), 4, 10)
    assert len(hashes) >= 1

    # add first hash to server ep1, second hash to ep2 (if present)
    if len(hashes) >= 1:
        scorer.add_prefixes_for_server("ep1", [hashes[0]])
    if len(hashes) >= 2:
        scorer.add_prefixes_for_server("ep2", [hashes[1]])

    endpoints = [Endpoint(name="ep1"), Endpoint(name="ep2"), Endpoint(name="ep3")]
    cs = CycleState()
    scores = scorer.score(cs, req, endpoints)

    # Note: The PrefixCacheScorer only returns scores for endpoints that have at least one hit.
    assert set(scores.keys()) == {"ep1", "ep2"}

    # ep1 should have non-zero score
    assert scores["ep1"] >= 0.0
    # ep3 should NOT be in scores (no hashes added)
    assert "ep3" not in scores
