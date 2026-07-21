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

from py_inference_scheduler.framework import CycleState, Endpoint, LLMRequest
from py_inference_scheduler.plugins.scorers.prefix_plugin import (
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
    # prepare a request with 3 blocks
    body = "abcdefghijkl"
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)

    # compute hashes using same logic
    hashes = _hash_prompt_bytes(req.target_model, body.encode("utf-8"), 4, 10)
    assert len(hashes) == 3

    # ep1 holds 2 of 3 blocks (above the half-coverage threshold), ep2 holds 1.
    scorer.add_prefixes_for_server("ep1", hashes[:2])
    scorer.add_prefixes_for_server("ep2", [hashes[2]])

    endpoints = {
        "ep1": Endpoint(name="ep1"),
        "ep2": Endpoint(name="ep2"),
        "ep3": Endpoint(name="ep3"),
    }
    cs = CycleState()
    scores = scorer.score(cs, req, endpoints)

    # Note: when the best match covers at least half the blocks, the scorer
    # only returns scores for endpoints that have at least one hit.
    assert set(scores.keys()) == {"ep1", "ep2"}

    # scores are normalized hit fractions
    assert scores["ep1"] == 2.0 / 3.0
    assert scores["ep2"] == 1.0 / 3.0
    # ep3 should NOT be in scores (no hashes added)
    assert "ep3" not in scores


def test_prefix_cache_scorer_weak_match_falls_back_to_least_loaded():
    """With a threshold set, a best match covering less than it is treated as
    a novel prompt."""
    scorer = PrefixCacheScorer(block_size=4, max_prefix_blocks=10, min_match_ratio=0.5)
    # 6 blocks
    body = "abcdefghijklmnopqrstuvwx"
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)
    hashes = _hash_prompt_bytes(req.target_model, body.encode("utf-8"), 4, 10)
    assert len(hashes) == 6

    # ep1 holds only 2 of 6 blocks -> best coverage 1/3, below the half threshold.
    scorer.add_prefixes_for_server("ep1", hashes[:2])

    endpoints = {"ep1": Endpoint(name="ep1"), "ep2": Endpoint(name="ep2")}
    scores = scorer.score(CycleState(), req, endpoints)

    # Fallback routes to the least-loaded server (fewest indexed hashes):
    # ep2 has nothing cached, so it wins outright; ep1's partial hits are zeroed.
    assert scores == {"ep1": 0.0, "ep2": 1.0}


def test_prefix_cache_scorer_half_coverage_is_not_novel():
    """Coverage exactly at the threshold stays on the normalized path (strict <)."""
    scorer = PrefixCacheScorer(block_size=4, max_prefix_blocks=10, min_match_ratio=0.5)
    # 4 blocks
    body = "abcdefghijklmnop"
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)
    hashes = _hash_prompt_bytes(req.target_model, body.encode("utf-8"), 4, 10)
    assert len(hashes) == 4

    # ep1 holds 2 of 4 blocks: max_score == total/2, so no fallback.
    scorer.add_prefixes_for_server("ep1", hashes[:2])
    # ep2 is completely idle; under fallback it would have scored 1.0.
    endpoints = {"ep1": Endpoint(name="ep1"), "ep2": Endpoint(name="ep2")}

    scores = scorer.score(CycleState(), req, endpoints)

    assert scores == {"ep1": 0.5}


def test_prefix_cache_scorer_min_match_ratio_is_configurable():
    """min_match_ratio moves the novel-prompt threshold in both directions."""
    body = "abcdefghijklmnopqrstuvwx"  # 6 blocks at block_size=4
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)
    endpoints = {"ep1": Endpoint(name="ep1"), "ep2": Endpoint(name="ep2")}

    def scorer_with(ratio: float) -> PrefixCacheScorer:
        s = PrefixCacheScorer(block_size=4, max_prefix_blocks=10, min_match_ratio=ratio)
        hashes = _hash_prompt_bytes(req.target_model, body.encode("utf-8"), 4, 10)
        assert len(hashes) == 6
        # ep1 holds 2 of 6 blocks: 1/3 coverage.
        s.add_prefixes_for_server("ep1", hashes[:2])
        return s

    # A threshold below the coverage keeps the normalized path.
    lenient = scorer_with(0.25)
    assert lenient.score(CycleState(), req, endpoints) == {"ep1": 2.0 / 6.0}

    # Raising it above the coverage forces the least-loaded fallback.
    strict = scorer_with(0.4)
    assert strict.score(CycleState(), req, endpoints) == {"ep1": 0.0, "ep2": 1.0}


def test_prefix_cache_scorer_default_ratio_never_falls_back_on_partial_match():
    """With min_match_ratio unset (default 0.0) the coverage threshold is
    inert: any endpoint with hits scores its normalized fraction, no matter
    how sparse the match, and idle pods get no least-loaded boost."""
    scorer = PrefixCacheScorer(block_size=4, max_prefix_blocks=10)
    body = "abcdefghijklmnopqrstuvwx"  # 6 blocks
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)
    hashes = _hash_prompt_bytes(req.target_model, body.encode("utf-8"), 4, 10)
    assert len(hashes) == 6

    # ep1: 2 of 6 blocks; ep2: a single block.
    scorer.add_prefixes_for_server("ep1", hashes[:2])
    scorer.add_prefixes_for_server("ep2", [hashes[2]])

    endpoints = {
        "ep1": Endpoint(name="ep1"),
        "ep2": Endpoint(name="ep2"),
        "ep3": Endpoint(name="ep3"),
    }
    scores = scorer.score(CycleState(), req, endpoints)

    # Standard prefix routing: normalized fractions, idle ep3 absent.
    assert scores == {"ep1": 2.0 / 6.0, "ep2": 1.0 / 6.0}


def test_prefix_cache_scorer_novel_prompt_ties_between_least_loaded():
    """With no prefix hits at all, every least-loaded pod scores 1.0."""
    scorer = PrefixCacheScorer(block_size=4, max_prefix_blocks=10)
    body = "abcdefghijkl"
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)

    # ep_busy has unrelated cached blocks; ep_a and ep_b are both empty.
    scorer.add_prefixes_for_server("ep_busy", [111, 222, 333])

    endpoints = {
        "ep_busy": Endpoint(name="ep_busy"),
        "ep_a": Endpoint(name="ep_a"),
        "ep_b": Endpoint(name="ep_b"),
    }
    scores = scorer.score(CycleState(), req, endpoints)

    # Both idle pods tie at 1.0; fallback scores are exact 0/1, never
    # divided by the block count.
    assert scores == {"ep_busy": 0.0, "ep_a": 1.0, "ep_b": 1.0}


def test_prefix_cache_scorer_fallback_then_affinity_develops():
    """After a novel prompt is routed via fallback, pre_request records its
    prefixes so a repeat of the same prompt scores the chosen server highest."""
    scorer = PrefixCacheScorer(block_size=4, max_prefix_blocks=10)
    body = "abcdefghijkl"
    req = LLMRequest(request_id="r1", target_model="m", headers={}, body=body)
    endpoints = {"ep1": Endpoint(name="ep1"), "ep2": Endpoint(name="ep2")}

    cs = CycleState()
    first = scorer.score(cs, req, endpoints)
    # Novel prompt: both pods are empty, so both tie as least-loaded.
    assert first == {"ep1": 1.0, "ep2": 1.0}

    # The router picks ep1; pre_request stores the hashes from cycle_state.
    scorer.pre_request(cs, req, endpoints["ep1"])

    repeat = scorer.score(CycleState(), req, endpoints)
    # Full coverage on ep1 -> normalized path, ep1 scores a perfect 1.0 and
    # ep2 (no hits) is absent.
    assert repeat == {"ep1": 1.0}
