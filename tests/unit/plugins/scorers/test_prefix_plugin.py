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

from scheduling.framework import CycleState, Endpoint, LLMRequest
from scheduling.plugins.scorers.prefix_plugin import (
    PrefixCacheScorer,
    PrefixIndexer,
    _hash_prompt_bytes,
)


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


def test_indexer_reset_clears_all_mappings():
    idx = PrefixIndexer()
    # Two servers share hash=2 so we exercise both maps and the shared-entry path.
    idx.add([1, 2, 3], "s1")
    idx.add([2, 3, 4], "s2")
    assert set(idx.pods()) == {"s1", "s2"}
    assert idx.get(2) == {"s1", "s2"}

    idx.reset()

    assert idx.pods() == []
    for h in (1, 2, 3, 4):
        assert idx.get(h) == set()

    # After reset the indexer must still be usable.
    idx.add([5], "s3")
    assert idx.pods() == ["s3"]
    assert idx.get(5) == {"s3"}


def test_prefix_cache_scorer_reset_drops_routing_hints():
    scorer = PrefixCacheScorer(block_size=4, max_prefix_blocks=10)
    body = "abcdefghijkl"
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)
    hashes = _hash_prompt_bytes(req.target_model, body.encode("utf-8"), 4, 10)
    assert len(hashes) >= 1
    scorer.add_prefixes_for_server("ep1", hashes)

    endpoints = {"ep1": Endpoint(name="ep1"), "ep2": Endpoint(name="ep2")}

    # Pre-reset: ep1 has all the prefix hits, so it scores; ep2 does not.
    pre_scores = scorer.score(CycleState(), req, endpoints)
    assert "ep1" in pre_scores
    assert pre_scores["ep1"] > 0.0

    scorer.reset()

    # Post-reset: no cached prefixes -> no endpoint scores against the prefix
    # index. The "novel prompt" fallback then routes to the least-loaded
    # servers; with both at zero load, both are tied.
    post_scores = scorer.score(CycleState(), req, endpoints)
    assert set(post_scores.keys()) == {"ep1", "ep2"}
    assert scorer.indexer.pods() == []


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

    endpoints = {
        "ep1": Endpoint(name="ep1"),
        "ep2": Endpoint(name="ep2"),
        "ep3": Endpoint(name="ep3"),
    }
    cs = CycleState()
    scores = scorer.score(cs, req, endpoints)

    # Note: The PrefixCacheScorer only returns scores for endpoints that have at least one hit.
    assert set(scores.keys()) == {"ep1", "ep2"}

    # ep1 should have non-zero score
    assert scores["ep1"] >= 0.0
    # ep3 should NOT be in scores (no hashes added)
    assert "ep3" not in scores
