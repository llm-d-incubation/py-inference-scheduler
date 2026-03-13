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

from typing import Any, Dict, List, Optional, Sequence, Set, Mapping
from ...framework import Endpoint, CycleState, LLMRequest, register_scorer
import hashlib
import json
from collections import OrderedDict 

class PrefixIndexer:
    """Simple in-memory indexer: maps block-hash -> set of server names.
    Also keep server -> set(hashes) for efficient removal.
    """

    def __init__(self, lru_capacity_per_server = 31250) -> None:
        self._hash_to_servers: Dict[int, Set[str]] = {}
        self._server_to_hashes: Dict[str, OrderedDict[int]] = {}
        self._lru_capacity_per_server = lru_capacity_per_server

    def add(self, hashes: Sequence[int], server: str) -> None:
        if server not in self._server_to_hashes:
            self._server_to_hashes[server] = OrderedDict()
        for h in hashes:
            if h not in self._hash_to_servers:
                self._hash_to_servers[h] = set()
            self._hash_to_servers[h].add(server)
            self._server_to_hashes[server][h] = None
       
        # Evict oldest entries
        while len(self._server_to_hashes[server]) > self._lru_capacity_per_server:
            old_h, _ = self._server_to_hashes[server].popitem(last=False)
            servs = self._hash_to_servers.get(old_h)
            if servs:
                servs.discard(server)
                if not servs:
                    self._hash_to_servers.pop(old_h, None)
        

    def get(self, h: int) -> Set[str]:
        return set(self._hash_to_servers.get(h, set()))

    def remove_server(self, server: str) -> None:
        hashes = self._server_to_hashes.pop(server, set())
        for h in hashes:
            servs = self._hash_to_servers.get(h)
            if not servs:
                continue
            servs.discard(server)
            if not servs:
                # drop empty entry
                self._hash_to_servers.pop(h, None)

    def pods(self) -> List[str]:
        return list(self._server_to_hashes.keys())


def _get_user_input_bytes(body: Any) -> Optional[bytes]:
    if body is None:
        return None
    # If it's a plain string prompt
    if isinstance(body, str):
        return body.encode("utf-8")
    # If it looks like a completions dict
    try:
        return json.dumps(body, separators=(",", ":")).encode("utf-8")
    except Exception:
        print("Unable to serialize body to bytes for prefix hashing")
        return None


def _hash_prompt_bytes(target_model: Optional[str], body_bytes: bytes, block_size: int, max_prefix_blocks: int) -> List[int]:
    if body_bytes is None:
        return []
    if len(body_bytes) < block_size:
        return []
    if len(body_bytes) > block_size * max_prefix_blocks:
        body_bytes = body_bytes[: block_size * max_prefix_blocks]

    res: List[int] = []
    # initial prev is model name (if any)
    prev = hashlib.sha256()
    if target_model:
        prev.update(target_model.encode("utf-8"))
    prev_digest = prev.digest()

    for i in range(0, len(body_bytes) - block_size + 1, block_size):
        h = hashlib.sha256()
        h.update(body_bytes[i : i + block_size])
        h.update(prev_digest)
        digest = h.digest()
        # use first 8 bytes as 64-bit hash
        val = int.from_bytes(digest[:8], "little")
        res.append(val)
        prev_digest = digest
    return res


@register_scorer("prefix_cache")
class PrefixCacheScorer:
    """A scorer that computes prefix-block hashes from the request and
    scores endpoints by fraction of blocks present in the local indexer for
    that endpoint (server name).

    Usage: instantiate once and pass as a ScorerPlugin to a profile. The
    returned score is in [0,1].
    """

    def __init__(self, block_size: int = 64, max_prefix_blocks: int = 256, lru_capacity_per_server = 31250) -> None:
        self.block_size = block_size
        self.max_prefix_blocks = max_prefix_blocks
        self.indexer = PrefixIndexer()

    def score(self, cycle_state: CycleState, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Dict[str, float]:
        # normalize endpoints to mapping name->Endpoint
        if isinstance(endpoints, Mapping):
            eps = endpoints
        else:
            eps = {e.name: e for e in endpoints}

        body_bytes = _get_user_input_bytes(request.body)
        hashes = _hash_prompt_bytes(request.target_model, body_bytes or b"", self.block_size, self.max_prefix_blocks)
        # store in cycle_state for potential use elsewhere
        cycle_state.set("prefix_hashes", hashes)

        total = len(hashes)
        if total == 0:
            return {name: 0.0 for name in eps.keys()}

        scores: Dict[str, float] = {}
        for h in hashes:
            servs = self.indexer.get(h)
            for name in servs:
                if name not in eps:
                    continue
                if name not in scores:
                    scores[name] = 1.0
                else:
                    scores[name] += 1
        
        # in the event of a novel prompt with no matching prefixes, route to the smallest loaded server(s)
        if len(scores) == 0:
            min_count = min([ len(self.indexer._server_to_hashes.get(name, {})) for name in eps.keys() ])
            for name in eps.keys():
                if len(self.indexer._server_to_hashes.get(name, {})) == min_count:
                    scores[name] = 1.0

        for name, score in scores.items():
            scores[name] = float(scores[name]) / float(total)

        return scores

    def pre_request(self, cycle_state: CycleState, request: LLMRequest, selected_endpoint: Endpoint) -> None:
        hashes = cycle_state.get("prefix_hashes")
        if hashes is not None:
            self.add_prefixes_for_server(selected_endpoint.name, hashes)
        else:
            print("Warning: prefix_hashes not found in cycle_state in pre_request")
            body_bytes = _get_user_input_bytes(request.body)
            hashes = _hash_prompt_bytes(request.target_model, body_bytes or b"", self.block_size, self.max_prefix_blocks)
            self.add_prefixes_for_server(selected_endpoint.name, hashes)

    # Helper to simulate the PreRequest behaviour in the upstream plugin.
    def add_prefixes_for_server(self, server_name: str, hashes: Sequence[int]) -> None:
        self.indexer.add(hashes, server_name)

    def remove_server(self, server_name: str) -> None:
        self.indexer.remove_server(server_name)

